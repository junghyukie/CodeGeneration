#!/usr/bin/env python3
"""
Build the ``calibration_MBPP`` split for the
``ankhanhtran02/CL4Code-executable-datasets`` Hugging Face dataset.

The split holds MBPP samples for the 9 supported languages and follows the
standard repo schema (``index``, ``language``, ``instruction``, ``solution``,
``test``) and the lowercase ``language`` convention used by ``preprocess.py``.

Sources
-------
Python  -> ``Muennighoff/mbpp`` config ``sanitized`` (split ``test``)
    * index       = task_id (kept as-is, stringified for the schema)
    * solution    = code
    * instruction = the prompt with "Write a[ python] function" rewritten to
      "Write a python function `<signature>`", where <signature> is the last
      "\\n"-separated line of ``code`` containing a ``def``.
    * test        = test_imports + a ``def check(<fn>): ...`` / ``check(<fn>)``
      harness built from test_list (the test_McEval Python convention), where
      <fn> is the function name parsed from the signature.

Other 8 languages -> ``nuprl/MultiPL-E`` config ``mbpp-<ext>`` (split ``test``)
    * index       = name (kept as-is)
    * solution    = None
    * instruction = the doc comment with "Write a <x> function" rewritten to
      "Write a {MAPPING[language]} function `<signature>`", where <signature>
      is the last non-empty line of the prompt and the comment marker is removed.
    * test        = the MultiPL-E ``tests`` with the single leading ``}`` (which
      closes the prompt-opened function body) removed, matching the test_McEval
      format where a test harness follows a complete solution.

Script-based datasets are loaded from their datasets-server auto-converted
parquet files, because recent ``datasets`` releases dropped loading-script
support.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, Features, Value, load_dataset
from huggingface_hub import HfApi


THIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = THIS_DIR / "output"

DEFAULT_REPO_ID = "ankhanhtran02/CL4Code-executable-datasets"
SPLIT = "calibration_MBPP"

DATA_FILE_SUFFIXES = {".arrow", ".csv", ".json", ".jsonl", ".parquet"}

# Standard repo schema (mirrors STANDARD_FEATURES in preprocess.py).
STANDARD_FEATURES = Features(
    {
        "index": Value("string"),
        "language": Value("string"),
        "instruction": Value("string"),
        "solution": Value("string"),
        "test": Value("string"),
    }
)

PYTHON = "python"

# repo 'language' value (lowercase convention) -> display name used inside the
# instruction documentation ("Write a {display} function ...").
MAPPING = {
    "python": "python",
    "cpp": "CPP",
    "swift": "Swift",
    "rust": "Rust",
    "csharp": "C#",
    "java": "Java",
    "php": "PHP",
    "typescript": "TypeScript",
    "shell": "Shell",
}

# Non-python repo 'language' value -> MultiPL-E source-file extension.
LANGUAGE_EXTENSIONS = {
    "cpp": "cpp",
    "swift": "swift",
    "rust": "rs",
    "csharp": "cs",
    "java": "java",
    "php": "php",
    "typescript": "ts",
    "shell": "sh",
}

_DEF_PATTERN = re.compile(r"\bdef\b")
_DEF_NAME_PATTERN = re.compile(r"\bdef\s+(\w+)")
# "Write a function" / "Write a python function" / "Write a cppthon function" ...
_WRITE_FUNCTION_PATTERN = re.compile(r"Write\s+an?\s+(?:\S+\s+)?function", re.IGNORECASE)
# A single leading "}" (optionally indented / preceded by blank lines).
_LEADING_BRACE_PATTERN = re.compile(r"\A[ \t\r\n]*\}[ \t]*\n?")


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN")


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _load_parquet(repo: str, config: str, split: str = "test") -> Dataset:
    """Load one config/split from the datasets-server auto-converted parquet files.

    Version-proof: works even though ``datasets`` dropped loading-script support
    for repos such as Muennighoff/mbpp and nuprl/MultiPL-E.
    """
    url = (
        f"https://huggingface.co/datasets/{repo}/resolve/"
        f"refs%2Fconvert%2Fparquet/{config}/{split}/0000.parquet"
    )
    return load_dataset("parquet", data_files=url, split="train", token=_hf_token())


def _row(index, language, instruction, solution, test) -> dict[str, Any]:
    return {
        "index": _as_str(index),
        "language": language,
        "instruction": _as_str(instruction),
        "solution": _as_str(solution),
        "test": _as_str(test),
    }


# --------------------------------------------------------------------------- #
# Instruction helpers
# --------------------------------------------------------------------------- #
def _extract_function_signature(code: str | None) -> str | None:
    """Return the last "\\n"-separated line of ``code`` that contains a ``def``.

    e.g. ``def diff_even_odd(list1):`` from a multi-line solution.
    """
    if code is None:
        return None
    signature_line: str | None = None
    for line in str(code).split("\n"):
        if _DEF_PATTERN.search(line):
            signature_line = line.strip()
    return signature_line


def _function_name(signature: str | None) -> str | None:
    if not signature:
        return None
    match = _DEF_NAME_PATTERN.search(signature)
    return match.group(1) if match else None


def _last_nonempty_line(text: str | None) -> str | None:
    if not text:
        return None
    lines = [line for line in str(text).split("\n") if line.strip()]
    return lines[-1].strip() if lines else None


def _strip_comment_marker(line: str) -> str:
    stripped = line.strip()
    for marker in ("///", "//", "#"):
        if stripped.startswith(marker):
            return stripped[len(marker):].strip()
    return stripped


def _insert_signature(doc_text: str, language: str, signature: str | None) -> str:
    """Rewrite "Write a[n] [<x>] function" -> "Write a {display} function `<sig>`"."""
    display = MAPPING[language]
    if signature:
        replacement = f"Write a {display} function `{signature}`"
        new_text, count = _WRITE_FUNCTION_PATTERN.subn(lambda _m: replacement, doc_text, count=1)
        if count:
            return new_text
        # No "Write a ... function" phrase to anchor on: keep a usable instruction.
        if not doc_text.strip():
            return replacement
    return doc_text


def _doc_from_prompt(prompt: str) -> str:
    """Return the (comment-stripped) documentation line of a MultiPL-E prompt."""
    for line in str(prompt).split("\n"):
        if _WRITE_FUNCTION_PATTERN.search(line):
            return _strip_comment_marker(line)
    return ""


# --------------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------------- #
def _build_python_test(func_name: str | None, test_imports, test_list) -> str:
    """Build a test_McEval-style Python harness from MBPP test_imports/test_list."""
    lines: list[str] = []
    lines.extend(test_imports or [])
    asserts = list(test_list or [])
    if func_name:
        lines.append(f"def check({func_name}):")
        lines.extend(f"    {assertion}" for assertion in asserts)
        lines.append("")
        lines.append(f"check({func_name})")
    else:
        lines.extend(asserts)
    return "\n".join(lines)


def _transform_multipl_e_test(tests: str) -> str:
    """Drop the single leading ``}`` that closes the prompt-opened function body."""
    return _LEADING_BRACE_PATTERN.sub("", tests, count=1)


# --------------------------------------------------------------------------- #
# Row builders
# --------------------------------------------------------------------------- #
def _build_python_rows() -> list[dict[str, Any]]:
    dataset = _load_parquet("Muennighoff/mbpp", "sanitized")
    rows = []
    for sample in dataset:
        code = sample["code"]
        signature = _extract_function_signature(code)
        func_name = _function_name(signature)
        # Skip problems whose function is itself named "check": it collides with
        # the test_McEval `check(<fn>)` wrapper, producing a non-runnable test.
        if func_name == "check":
            continue
        instruction = _insert_signature(sample["prompt"] or "", PYTHON, signature)
        test = _build_python_test(func_name, sample.get("test_imports"), sample.get("test_list"))
        rows.append(
            _row(
                index=sample["task_id"],
                language=PYTHON,
                instruction=instruction,
                solution=code,
                test=test,
            )
        )
    return rows


def _build_multipl_e_rows(language: str, extension: str) -> list[dict[str, Any]]:
    dataset = _load_parquet("nuprl/MultiPL-E", f"mbpp-{extension}")
    rows = []
    for sample in dataset:
        signature = _last_nonempty_line(sample["prompt"])
        doc = _doc_from_prompt(sample["prompt"])
        if doc:
            instruction = _insert_signature(doc, language, signature)
        else:
            instruction = f"Write a {MAPPING[language]} function `{signature}`" if signature else ""
        rows.append(
            _row(
                index=sample["name"],
                language=language,
                instruction=instruction,
                solution=None,
                test=_transform_multipl_e_test(sample["tests"]),
            )
        )
    return rows


def build_calibration_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    python_rows = _build_python_rows()
    print(f"  python (mbpp-sanitized): {len(python_rows)} samples")
    rows.extend(python_rows)

    for language, extension in LANGUAGE_EXTENSIONS.items():
        language_rows = _build_multipl_e_rows(language, extension)
        print(f"  {language} (mbpp-{extension}): {len(language_rows)} samples")
        rows.extend(language_rows)

    return rows


# --------------------------------------------------------------------------- #
# Hub upload
# --------------------------------------------------------------------------- #
def _is_split_data_file(repo_file: str, split: str) -> bool:
    path = Path(repo_file)
    return path.name.startswith(f"{split}-") and path.suffix in DATA_FILE_SUFFIXES


def _clear_hub_split(repo_id: str, split: str) -> None:
    """Remove existing Hub data files for ``split`` so a re-run does not duplicate shards."""
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


def _write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and push the calibration_MBPP split across 9 languages to the "
            "CL4Code executable dataset on the Hugging Face Hub."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repo id.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the split and write the JSONL file, but do not delete or upload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Building split {SPLIT!r}:")
    rows = build_calibration_rows()
    dataset = Dataset.from_list(rows, features=STANDARD_FEATURES)

    jsonl_path = OUTPUT_DIR / f"{SPLIT}.jsonl"
    _write_jsonl(rows, jsonl_path)
    print(f"Built {len(dataset)} samples for {SPLIT!r}; wrote {jsonl_path}")

    if args.dry_run:
        print("Dry run enabled; skipping split deletion and upload.")
        return

    _clear_hub_split(args.repo_id, SPLIT)
    dataset.push_to_hub(
        args.repo_id,
        split=SPLIT,
        token=_hf_token(),
        commit_message=f"Add {SPLIT} split (MBPP) across {len(MAPPING)} languages",
    )
    print(f"Uploaded {len(dataset)} samples to {args.repo_id}/{SPLIT}")


if __name__ == "__main__":
    main()