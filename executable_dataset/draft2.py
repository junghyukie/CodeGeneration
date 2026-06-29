#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ID = "ankhanhtran02/CL4Code-executable-datasets"
DEFAULT_SPLIT = "train_OSS_Instruct"
DEFAULT_LANGUAGE = "python"
DEFAULT_N = 4700
DEFAULT_RANDOM_SEED = 42
DEFAULT_JSONL_PATH = THIS_DIR / "output" / f"{DEFAULT_SPLIT}_all.jsonl"
DATA_FILE_SUFFIXES = {".arrow", ".csv", ".json", ".jsonl", ".parquet"}


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN")


def _write_jsonl(dataset: Dataset, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(input_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object on line {line_number} in {input_path}")
            rows.append(value)
    return rows


def _selected_language_indices(
    rows: list[dict[str, Any]],
    language: str,
    n: int,
    random_seed: int,
) -> set[int]:
    if n < 0:
        raise ValueError("--n must be non-negative.")

    language_indices = [
        index for index, row in enumerate(rows) if row.get("language") == language
    ]
    if n > len(language_indices):
        raise ValueError(
            f"Requested n={n}, but only found {len(language_indices)} samples "
            f"with language={language!r}."
        )

    rng = random.Random(random_seed)
    return set(rng.sample(language_indices, n))


def _build_replacement_rows(
    rows: list[dict[str, Any]],
    language: str,
    n: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    sampled_language_indices = _selected_language_indices(
        rows=rows,
        language=language,
        n=n,
        random_seed=random_seed,
    )
    return [
        row
        for index, row in enumerate(rows)
        if row.get("language") != language or index in sampled_language_indices
    ]


def _is_split_data_file(repo_file: str, split: str) -> bool:
    path = Path(repo_file)
    return path.name.startswith(f"{split}-") and path.suffix in DATA_FILE_SUFFIXES


def _clear_hub_split(repo_id: str, split: str) -> None:
    token = _hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN not set; cannot authenticate to Hugging Face Hub.")

    api = HfApi(token=token)
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    split_files = [repo_file for repo_file in repo_files if _is_split_data_file(repo_file, split)]

    if not split_files:
        print(f"No existing Hub data files found for split={split!r}.")
        return

    for repo_file in split_files:
        api.delete_file(
            path_in_repo=repo_file,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Remove existing {split} data file: {repo_file}",
        )
        print(f"Removed existing Hub file: {repo_file}")


def build_replacement_split(
    repo_id: str,
    split: str,
    jsonl_path: Path,
    language: str,
    n: int,
    random_seed: int,
) -> Dataset:
    source_dataset = load_dataset(repo_id, split=split, token=_hf_token())
    if len(source_dataset) == 0:
        raise ValueError(f"No samples found in split={split!r}.")

    _write_jsonl(source_dataset, jsonl_path)
    rows = _read_jsonl(jsonl_path)
    replacement_rows = _build_replacement_rows(
        rows=rows,
        language=language,
        n=n,
        random_seed=random_seed,
    )
    if not replacement_rows:
        raise ValueError("Replacement split would be empty.")

    return Dataset.from_list(replacement_rows, features=source_dataset.features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Archive a Hugging Face dataset split to JSONL, remove the split on the Hub, "
            "then re-upload all non-target-language samples plus n random target-language samples."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repo id.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split to replace.")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help="Language value to sample from.")
    parser.add_argument("--jsonl-path", type=Path, default=DEFAULT_JSONL_PATH, help="Where to save all split rows.")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of target-language rows to keep.")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed for sampling.")
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Optional Hub commit message.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the replacement dataset and JSONL file, but do not delete or upload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replacement_dataset = build_replacement_split(
        repo_id=args.repo_id,
        split=args.split,
        jsonl_path=args.jsonl_path,
        language=args.language,
        n=args.n,
        random_seed=args.random_seed,
    )

    language_count = sum(1 for row in replacement_dataset if row.get("language") == args.language)
    other_count = len(replacement_dataset) - language_count
    print(f"Saved all original samples to: {args.jsonl_path}")
    print(
        f"Built replacement split with {len(replacement_dataset)} samples: "
        f"{language_count} {args.language}, {other_count} other-language samples."
    )

    if args.dry_run:
        print("Dry run enabled; skipping split deletion and upload.")
        return

    _clear_hub_split(args.repo_id, args.split)
    replacement_dataset.push_to_hub(
        args.repo_id,
        split=args.split,
        token=_hf_token(),
        commit_message=args.commit_message
        or (
            f"Replace {args.split} with all non-{args.language} samples "
            f"and {args.n} {args.language} samples"
        ),
    )
    print(f"Uploaded {len(replacement_dataset)} samples to {args.repo_id}/{args.split}")


if __name__ == "__main__":
    main()
