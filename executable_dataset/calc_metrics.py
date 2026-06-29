#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from concurrent.futures import ProcessPoolExecutor, BrokenExecutor
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from evaluate import SUPPORTED_LANGUAGES, evaluate, stop_docker_container
from tqdm import tqdm
import time

DEFAULT_DATASET_REPO = "ankhanhtran02/CL4Code-executable-datasets"


def _eval_candidate(args: tuple[str, str, str]) -> dict:
    language, candidate, test = args
    return evaluate(language, candidate, test)

def _infer_language(file_name: str) -> str:
	name_lower = file_name.lower()
	for language in SUPPORTED_LANGUAGES:
		if language.lower() in name_lower:
			return language
	return ""


def _convert_online_cl_row(row: dict) -> dict:
	instance = row.get("Instance", {})
	ground_truth = instance.get("ground_truth", "") or None
	return {
		"source": instance.get("sentence", ""),
		"ground-truth": ground_truth,
		"prediction": row.get("Predictions", []),
	}


def _load_instruction_tests(dataset_repo: str, language: str, split: str = "test_McEval") -> dict[str, str]:
	dataset = load_dataset(dataset_repo, split=split)
	if language:
		dataset = dataset.filter(lambda row: row.get("language") == language)
	return {row["instruction"]: row["test"] for row in dataset}


def _download_prediction(repo_id: str, repo_path: str, output_dir: Path) -> Path:
	# 1. Download to the default HF cache
	local_cache_path = hf_hub_download(repo_id=repo_id, filename=repo_path)
	
	print(f"Downloaded {repo_path} from {repo_id} to cache at {local_cache_path}")
	
	# 2. Define your specific output path
	output_path = output_dir / repo_path

	# 3. Ensure the parent directories exist (e.g., output_dir/subfolder/)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	
	# 4. Move the file from cache to your desired path
	# This effectively "deletes" it from the cache by moving it
	shutil.copy2(local_cache_path, output_path)
	
	return output_path


def _read_prediction_payload(path: Path) -> tuple[dict, bool]:
	text = path.read_text(encoding="utf-8")
	try:
		payload = json.loads(text)
		return payload, False
	except json.JSONDecodeError:
		rows = [json.loads(line) for line in text.splitlines() if line.strip()]
		return {"metrics": {}, "predictions": rows}, True


def _write_prediction_payload(path: Path, payload: dict, as_jsonl: bool) -> None:
	if as_jsonl:
		lines = [json.dumps(row, ensure_ascii=False) for row in payload.get("predictions", [])]
		path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
		return
	path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def add_tests_to_predictions(
	repo_id: str,
	repo_path: str,
	output_dir: Path,
	dataset_repo: str,
	row_preprocessor=None,
	split: str = "test_McEval",
) -> tuple[Path, int, int]:
	output_path = _download_prediction(repo_id, repo_path, output_dir)
	payload, _ = _read_prediction_payload(output_path)

	file_language = _infer_language(repo_path)
	payload["language"] = file_language

	predictions = payload.get("predictions", [])
	if row_preprocessor is not None:
		predictions = [row_preprocessor(row) for row in predictions]
		payload["predictions"] = predictions

	instruction_tests = {k.strip().strip("\n"): v for k, v in _load_instruction_tests(dataset_repo, file_language, split).items()}

	missing = 0
	for row in predictions:
		source = row.get("source", "").strip().strip("\n")
		test = instruction_tests.get(source)
		if test is None:
			missing += 1
		else:
			row["test"] = test

	payload["predictions"] = predictions
	_write_prediction_payload(output_path, payload, as_jsonl=False)
	return output_path, missing, len(predictions)


def execute_predictions(file_path: Path, max_workers: int | None = None) -> Path:
	print(f"#### Executing predictions in {file_path}... ####")
	payload, as_jsonl = _read_prediction_payload(file_path)
	language = payload.get("language") or _infer_language(file_path.name)
	if not language:
		raise ValueError("Could not infer language from file name or payload.")
	print(f"Inferred language: {language}")

	predictions = payload.get("predictions", [])
	metrics = payload.get("metrics", {})

	all_predictions = 0
	total_passed = 0
	all_passed_samples = 0
	all_failed_samples = 0
	backend_logged = False

	use_pool = True
	executor = ProcessPoolExecutor(max_workers=max_workers)
	try:
		for row in tqdm(predictions):
			raw = row.get("prediction") or row.get("predictions") or []
			candidates = [raw] if isinstance(raw, str) else raw
			test = row.get("test", "")

			args = [(language, candidate, test) for candidate in candidates]

			if use_pool:
				try:
					results = list(executor.map(_eval_candidate, args))
				except (BrokenExecutor, Exception) as exc:
					print(f"\nProcess pool broke ({exc}); switching to sequential execution.")
					use_pool = False
					try:
						executor.shutdown(wait=False)
					except Exception:
						pass
					results = [_eval_candidate(arg) for arg in args]
			else:
				results = [_eval_candidate(arg) for arg in args]

			if not backend_logged and results:
				backend = "Docker" if results[0].get("use_docker") else "local"
				print(f"Evaluator backend: {backend}")
				backend_logged = True

			passed_flags = [int(bool(r.get("test_passed"))) for r in results]
			stderr_logs = ["" if p else r.get("stderr", "") for p, r in zip(passed_flags, results)]

			num_passed = sum(passed_flags)
			row["passed"] = passed_flags
			row["stderr"] = stderr_logs
			row["num_passed"] = num_passed

			all_predictions += len(candidates)
			total_passed += num_passed
			if candidates and num_passed == len(candidates):
				all_passed_samples += 1
			if not candidates or num_passed == 0:
				all_failed_samples += 1
	finally:
		if use_pool:
			executor.shutdown(wait=True)

	stop_docker_container()

	metrics["num_samples"] = len(predictions)
	metrics["num_predictions"] = all_predictions
	metrics["total_passed_predictions"] = total_passed
	metrics["num_all_passed_samples"] = all_passed_samples
	metrics["num_all_failed_samples"] = all_failed_samples
	payload["metrics"] = metrics
	payload["predictions"] = predictions

	_write_prediction_payload(file_path, payload, as_jsonl=as_jsonl)
	return file_path


def _estimate_pass_at_k(num_samples: int, num_correct: list[int], k: int) -> np.ndarray:
	def estimator(n: int, c: int, k_val: int) -> float:
		if n - c < k_val:
			return 1.0
		return 1.0 - np.prod(1.0 - k_val / np.arange(n - c + 1, n + 1))

	return np.array([estimator(int(num_samples), int(c), k) for c in num_correct])


def calculate_pass_at_k(file_path: Path, num_samples: int = 5, ks: list[int] | None = None) -> dict:
	if ks is None:
		ks = [1, 5]
	payload, as_jsonl = _read_prediction_payload(file_path)
	metrics = payload.get("metrics", {})
	predictions = payload.get("predictions", [])

	num_correct = [row.get("num_passed", 0) for row in predictions]
	for k in ks:
		metrics[f"pass_at_{k}"] = float(_estimate_pass_at_k(num_samples, num_correct, k).mean()) if num_correct else 0.0

	payload["metrics"] = metrics

	_write_prediction_payload(file_path, payload, as_jsonl=as_jsonl)
	return metrics


def add_tests_to_local_predictions(
	file_path: Path,
	dataset_repo: str,
	row_preprocessor=None,
	split: str = "test_McEval",
) -> tuple[int, int]:
	payload, _ = _read_prediction_payload(file_path)

	file_language = _infer_language(str(file_path))
	payload["language"] = file_language

	predictions = payload.get("predictions", [])
	if row_preprocessor is not None:
		predictions = [row_preprocessor(row) for row in predictions]
		payload["predictions"] = predictions

	instruction_tests = {k.strip().strip("\n"): v for k, v in _load_instruction_tests(dataset_repo, file_language, split).items()}

	missing = 0
	for row in predictions:
		source = row.get("source", "").strip().strip("\n")
		test = instruction_tests.get(source)
		if test is None:
			missing += 1
		else:
			row["test"] = test

	payload["predictions"] = predictions
	_write_prediction_payload(file_path, payload, as_jsonl=False)
	return missing, len(predictions)


def run_local_pipeline(
	file_paths: list[Path],
	dataset_repo: str,
	num_samples: int = 5,
	max_workers: int | None = None,
	row_preprocessor=None,
	ks: list[int] | None = None,
	split: str = "test_McEval",
) -> None:
	for file_path in file_paths:
		print()
		missing, total = add_tests_to_local_predictions(
			file_path=file_path,
			dataset_repo=dataset_repo,
			row_preprocessor=row_preprocessor,
			split=split,
		)
		print(f"{file_path}: {missing}/{total} tests missing")
		execute_predictions(file_path, max_workers=max_workers)
		metrics = calculate_pass_at_k(file_path, num_samples=num_samples, ks=ks)
		print(f"{file_path}: {json.dumps(metrics, ensure_ascii=False)}")
		print("---------------------")


def run_full_pipeline(
	repo_id: str,
	prediction_paths: list[str],
	output_dir: Path,
	dataset_repo: str,
	num_samples: int = 5,
	max_workers: int | None = None,
	row_preprocessor=None,
	ks: list[int] | None = None,
	split: str = "test_McEval",
) -> None:
	for repo_path in prediction_paths:
		print()
		output_path, missing, total = add_tests_to_predictions(
			repo_id=repo_id,
			repo_path=repo_path,
			output_dir=output_dir,
			dataset_repo=dataset_repo,
			row_preprocessor=row_preprocessor,
			split=split,
		)
		print(f"{output_path}: {missing}/{total} tests missing")
		execute_predictions(output_path, max_workers=max_workers)
		metrics = calculate_pass_at_k(output_path, num_samples=num_samples, ks=ks)
		print(f"{output_path}: {json.dumps(metrics, ensure_ascii=False)}")
		print("---------------------")
		time.sleep(5)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Attach tests, execute predictions, and compute pass@k for HF prediction files.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument("--repo-id", required=True, help="Hugging Face repo id containing prediction files.")
	parser.add_argument(
		"--prediction-paths",
		nargs="+",
		required=True,
		help="File paths inside the repo to prediction JSON/JSONL files.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("output") / "predictions_with_tests",
		help="Directory to download and write updated prediction files.",
	)
	parser.add_argument(
		"--dataset-repo",
		default=DEFAULT_DATASET_REPO,
		help="Dataset repo containing test_McEval split.",
	)
	parser.add_argument("--num-samples", type=int, default=5, help="Number of samples per problem.")
	parser.add_argument("--ks", nargs="+", type=int, default=[1, 5], help="Values of k for pass@k computation.")
	parser.add_argument("--max-workers", type=int, default=None, help="Max parallel workers for candidate evaluation.")
	parser.add_argument("--split", default="test_McEval", help="Dataset split to load tests from.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	run_full_pipeline(
		repo_id=args.repo_id,
		prediction_paths=args.prediction_paths,
		output_dir=args.output_dir,
		dataset_repo=args.dataset_repo,
		num_samples=args.num_samples,
		max_workers=args.max_workers,
		ks=args.ks,
		split=args.split,
	)


if __name__ == "__main__":
	# main()
	langs = (
    "python",
    "cpp",
    "swift",
    "rust",
    "csharp",
    "java",
    "php",
    "typescript",
    "shell",
)
	num_samples = 5
	max_workers = 3  # shell runs locally; keep low to avoid OOM in WSL
	ks = [1, 5]
	split = "test_McEval"  

	repo_ids = [
		'ankhanhtran02/round2_disagree_explore_refined',
	]
	for repo_id in repo_ids:
		repo_name = repo_id.split("/")[-1]
		output_dir = Path("executable_dataset_2/output/") / repo_name
		prediction_paths = []
		for i in range(len(langs)):
			l = langs[i]
			prediction_paths.append(f"results-8-{l}-round2.json")
		try:
			run_full_pipeline(
				repo_id=repo_id,
				prediction_paths=prediction_paths,
				output_dir=output_dir,
				dataset_repo=DEFAULT_DATASET_REPO,
				num_samples=num_samples,
				max_workers=max_workers,
				ks=ks,
				split=split,
			)
		except Exception as e:
			print(f"Error during execution: {e}")
