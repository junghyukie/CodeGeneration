import argparse
import json
import os
import collections
import math
from typing import Any, Dict, List

from smoothbleu import compute_smooth_bleu


def load_eval_records_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    records: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as fp:
        for line_idx, line in enumerate(fp, start=1):
            s = line.strip()
            if not s:
                continue

            try:
                obj = json.loads(s)
            except json.JSONDecodeError as ex:
                raise ValueError(f"Invalid JSON at line {line_idx} in {jsonl_path}: {ex}") from ex

            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_idx} must be a JSON object.")

            required = ["task", "target", "prediction"]
            missing = [k for k in required if k not in obj]
            if missing:
                raise ValueError(f"Missing keys at line {line_idx}: {missing}")

            records.append(obj)

    return records


class BleuScorer:
    """Corpus BLEU implementation aligned with the provided reference."""

    @staticmethod
    def _get_ngrams(segment: List[str], max_order: int) -> collections.Counter:
        ngram_counts: collections.Counter = collections.Counter()
        for order in range(1, max_order + 1):
            for i in range(0, len(segment) - order + 1):
                ngram = tuple(segment[i : i + order])
                ngram_counts[ngram] += 1
        return ngram_counts

    def compute_bleu(
        self,
        reference_corpus: List[List[List[str]]],
        translation_corpus: List[List[str]],
        max_order: int = 4,
        smooth: bool = False,
    ) -> float:
        if not reference_corpus or not translation_corpus:
            return 0.0

        matches_by_order = [0] * max_order
        possible_matches_by_order = [0] * max_order
        reference_length = 0
        translation_length = 0

        for references, translation in zip(reference_corpus, translation_corpus):
            if not references or not translation or not any(references):
                continue

            non_empty_refs = [r for r in references if r]
            if not non_empty_refs:
                continue

            reference_length += min(len(r) for r in non_empty_refs)
            translation_length += len(translation)

            merged_ref_ngram_counts: collections.Counter = collections.Counter()
            for reference in non_empty_refs:
                merged_ref_ngram_counts |= self._get_ngrams(reference, max_order)

            translation_ngram_counts = self._get_ngrams(translation, max_order)
            overlap = translation_ngram_counts & merged_ref_ngram_counts

            for ngram in overlap:
                matches_by_order[len(ngram) - 1] += overlap[ngram]

            for order in range(1, max_order + 1):
                possible_matches = len(translation) - order + 1
                if possible_matches > 0:
                    possible_matches_by_order[order - 1] += possible_matches

        precisions = [0.0] * max_order
        for i in range(max_order):
            if smooth:
                precisions[i] = (matches_by_order[i] + 1.0) / (possible_matches_by_order[i] + 1.0)
            elif possible_matches_by_order[i] > 0:
                precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]

        if min(precisions) > 0:
            p_log_sum = sum((1.0 / max_order) * math.log(p) for p in precisions)
            geo_mean = math.exp(p_log_sum)
        else:
            geo_mean = 0.0

        if reference_length == 0:
            return 0.0

        ratio = float(translation_length) / reference_length
        if ratio > 1.0:
            bp = 1.0
        elif ratio > 0.0:
            bp = math.exp(1 - 1.0 / ratio)
        else:
            bp = 0.0

        return geo_mean * bp


bleu_scorer = BleuScorer()


def compute_bleu_corpus(preds: List[str], refs: List[str]) -> float:
    if not preds or not refs:
        return 0.0

    pred_tokens = [p.split() for p in preds]
    ref_tokens = [[r.split()] for r in refs]
    return float(bleu_scorer.compute_bleu(reference_corpus=ref_tokens, translation_corpus=pred_tokens))


def summarize_eval_records(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    per_task_refs: Dict[str, List[str]] = {}
    per_task_preds: Dict[str, List[str]] = {}
    per_task_total: Dict[str, int] = {}

    total = 0

    for rec in records:
        task = str(rec["task"])
        target = str(rec["target"])
        prediction = str(rec["prediction"])

        if task not in per_task_refs:
            per_task_refs[task] = []
            per_task_preds[task] = []
            per_task_total[task] = 0

        per_task_refs[task].append(target)
        per_task_preds[task].append(prediction)
        per_task_total[task] += 1

        total += 1

    per_task: Dict[str, Dict[str, Any]] = {}
    bleu_values: List[float] = []
    smooth_bleu_values: List[float] = []

    for task in sorted(per_task_refs.keys()):
        refs = per_task_refs[task]
        preds = per_task_preds[task]

        bleu = compute_bleu_corpus(preds=preds, refs=refs)
        smooth_bleu = float(compute_smooth_bleu([[r] for r in refs], preds, smooth=1))
        n = per_task_total[task]

        per_task[task] = {
            "n": n,
            "bleu": bleu,
            "smooth_bleu": smooth_bleu,
        }
        bleu_values.append(bleu)
        smooth_bleu_values.append(smooth_bleu)

    all_refs = [str(r["target"]) for r in records]
    all_preds = [str(r["prediction"]) for r in records]

    corpus_bleu = compute_bleu_corpus(preds=all_preds, refs=all_refs)
    corpus_smooth_bleu = float(compute_smooth_bleu([[r] for r in all_refs], all_preds, smooth=1))

    macro_bleu = float(sum(bleu_values) / len(bleu_values)) if bleu_values else 0.0
    macro_smooth_bleu = float(sum(smooth_bleu_values) / len(smooth_bleu_values)) if smooth_bleu_values else 0.0

    return {
        "total": total,
        "corpus_bleu": corpus_bleu,
        "corpus_smooth_bleu": corpus_smooth_bleu,
        "macro_bleu": macro_bleu,
        "macro_smooth_bleu": macro_smooth_bleu,
        "per_task": per_task,
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n=== Evaluation Metrics Summary ===")
    print(f"Total samples: {int(summary.get('total', 0))}")
    print(f"Corpus BLEU: {float(summary.get('corpus_bleu', 0.0)):.4f}")
    print(f"Corpus SmoothBLEU: {float(summary.get('corpus_smooth_bleu', 0.0)):.4f}")
    print(f"Macro BLEU (per-task average): {float(summary.get('macro_bleu', 0.0)):.4f}")
    print(f"Macro SmoothBLEU (per-task average): {float(summary.get('macro_smooth_bleu', 0.0)):.4f}")
    print("Per-task:")

    per_task = summary.get("per_task", {})
    for task in sorted(per_task.keys()):
        m = per_task[task]
        print(
            f"  {task:20s} "
            f"N={int(m.get('n', 0)):6d} "
            f"bleu={float(m.get('bleu', 0.0)):.4f} "
            f"smooth_bleu={float(m.get('smooth_bleu', 0.0)):.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute BLEU and SmoothBLEU from eval JSONL")
    parser.add_argument("--jsonl", type=str, required=True, help="Path to input eval JSONL file")
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional path to save metrics summary as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = load_eval_records_jsonl(args.jsonl)
    summary = summarize_eval_records(records=records)
    print_summary(summary)

    if args.output_json.strip():
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=2)
        print(f"\nSaved summary JSON to: {args.output_json}")


if __name__ == "__main__":
    main()
