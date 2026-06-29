from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from datasets import Dataset, DatasetDict, Features, Value, load_dataset
from huggingface_hub import HfApi


OUTPUT_DIR = Path(__file__).parent / "output"
DEFAULT_DUPLICATES_JSONL = OUTPUT_DIR / "exact_duplicates.jsonl"
DEFAULT_NEAR_DUPLICATES_JSONL = OUTPUT_DIR / "near_duplicates.jsonl"

LANGUAGE_MAPPINGS = {
  'CPP': 'cpp',
  'C++': 'cpp',
  'cpp': 'cpp',
  'Python': 'python',
  'python': 'python',
  'Swift': 'swift',
  'swift': 'swift',
  'Rust': 'rust',
  'rust': 'rust',
  'C#': 'csharp',
  'csharp': 'csharp',
  'Java': 'java',
  'java': 'java',
  'PHP': 'php',
  'php': 'php',
  'TypeScript': 'typescript',
  'typescript': 'typescript',
  'Shell':  'shell',
  'shell':  'shell',
}

LANGUAGE_COLUMNS = {
    "OSS-Instruct": "lang",
    "McEval-Instruct": "language",
    "McEval": "task_id",
}

INSTRUCTION_COLUMNS = {
    "OSS-Instruct": "problem",
    "McEval-Instruct": "instruction",
    "McEval": "instruction",
}

SOLUTION_COLUMNS = {
    "OSS-Instruct": "solution",
    "McEval-Instruct": "output",
    "McEval": None,
}

TEST_COLUMNS = {
    "OSS-Instruct": None,
    "McEval-Instruct": None,
    "McEval": "test",
}

INDEX_COLUMNS = {
    "OSS-Instruct": "index",
    "McEval-Instruct": "index",
    "McEval": "task_id",
}

LANGUAGES = {
    "cpp",
    "python",
    "swift",
    "rust",
    "csharp",
    "java",
    "php",
    "typescript",
    "shell",
}

STANDARD_FEATURES = Features(
    {
        "index": Value("string"),
        "language": Value("string"),
        "instruction": Value("string"),
        "solution": Value("string"),
        "test": Value("string"),
    }
)

_CODE_BLOCK_PATTERN = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)
_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")
_MULTI_BLANK_LINES = re.compile(r"\n{3,}")
_ZERO_WIDTH_CHARS = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_PLACEHOLDER_COMMENT_PATTERN = re.compile(
    r"(?im)^\s*(?:#|//|/\*+|\*+)?\s*(?:your code here|todo|fill in the missing code|implement here)\s*(?:\*/)?\s*$"
)
_CLASS_DEF_PATTERN = re.compile(r"\bclass\s+([a-z_][a-z0-9_]*)", re.IGNORECASE)
_FUNC_DEF_PATTERN = re.compile(r"\bdef\s+([a-z_][a-z0-9_]*)", re.IGNORECASE)
_FUNC_ARG_PATTERN = re.compile(r"\bdef\s+[a-z_][a-z0-9_]*\s*\((.*?)\)", re.IGNORECASE | re.DOTALL)
_IDENTIFIER_PATTERN = re.compile(r"\b[a-z_][a-z0-9_]*\b")
_PUNCTUATION_TO_DROP_PATTERN = re.compile(r"[():.,=]")
_IMPORT_LINE_PATTERN = re.compile(r"^\s*(?:from\s+[^\n]+\s+import\s+[^\n]+|import\s+[^\n]+)\s*$", re.MULTILINE)


def load_instructions(ds_list):
    instructions_list = []
    if "OSS-Instruct" in ds_list:
        ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
        for item in ds:
            instructions_list.append(
                {
                    "id": str(item["index"]),
                    "instruction": item["problem"],
                    "source": "OSS-Instruct",
                }
            )

    if "McEval-Instruct" in ds_list:
        ds = load_dataset("Multilingual-Multimodal-NLP/McEval-Instruct", split="train")
        for i, item in enumerate(ds):
            instructions_list.append(
                {
                    "id": str(i),
                    "instruction": item["instruction"],
                    "source": "McEval-Instruct",
                }
            )

    if "McEval" in ds_list:
        ds = load_dataset("Multilingual-Multimodal-NLP/McEval", "generation")
        for item in ds["test"]:
            instructions_list.append(
                {
                    "id": str(item["task_id"]),
                    "instruction": item["instruction"],
                    "source": "McEval",
                }
            )

    return Dataset.from_list(instructions_list)


def load_instruction_datasets(ds_list: list[str]) -> dict[str, Dataset]:
    datasets_by_source: dict[str, Dataset] = {}

    if "OSS-Instruct" in ds_list:
        datasets_by_source["OSS-Instruct"] = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")

    if "McEval-Instruct" in ds_list:
        ds = load_dataset("Multilingual-Multimodal-NLP/McEval-Instruct", split="train")
        if "index" not in ds.column_names:
            ds = ds.map(lambda _, idx: {"index": idx}, with_indices=True)
        datasets_by_source["McEval-Instruct"] = ds

    if "McEval" in ds_list:
        datasets_by_source["McEval"] = load_dataset("Multilingual-Multimodal-NLP/McEval", "generation")["test"]

    return datasets_by_source


def _normalize_prose_segment(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ZERO_WIDTH_CHARS.sub("", text)

    normalized_lines = []
    for line in text.split("\n"):
        stripped_line = _HORIZONTAL_WHITESPACE.sub(" ", line).strip()
        normalized_lines.append(stripped_line)

    normalized_text = "\n".join(normalized_lines)
    normalized_text = _MULTI_BLANK_LINES.sub("\n\n", normalized_text)
    return normalized_text.strip().lower()


def _normalize_code_block(block: str) -> str:
    opening_fence, _, remainder = block.partition("\n")
    language = opening_fence[3:].strip().lower()
    code = remainder[:-3] if remainder.endswith("```") else remainder

    code = unicodedata.normalize("NFKC", code)
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = _ZERO_WIDTH_CHARS.sub("", code)

    normalized_lines = [line.rstrip() for line in code.split("\n")]
    while normalized_lines and normalized_lines[0] == "":
        normalized_lines.pop(0)
    while normalized_lines and normalized_lines[-1] == "":
        normalized_lines.pop()

    normalized_code = "\n".join(normalized_lines).lower()
    if language:
        return f"```{language}\n{normalized_code}\n```"
    return f"```\n{normalized_code}\n```"


def normalize_instruction_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text)
    normalized_parts = []
    cursor = 0

    for match in _CODE_BLOCK_PATTERN.finditer(text):
        prose = text[cursor:match.start()]
        if prose:
            normalized_prose = _normalize_prose_segment(prose)
            if normalized_prose:
                normalized_parts.append(normalized_prose)

        normalized_parts.append(_normalize_code_block(match.group(0)))
        cursor = match.end()

    tail = text[cursor:]
    if tail:
        normalized_tail = _normalize_prose_segment(tail)
        if normalized_tail:
            normalized_parts.append(normalized_tail)

    normalized_text = "\n\n".join(normalized_parts).strip()
    return _MULTI_BLANK_LINES.sub("\n\n", normalized_text)

def _extract_import_tokens(code: str) -> tuple[str, list[str]]:
    import_tokens = []
    kept_lines = []

    for line in code.split("\n"):
        stripped_line = line.strip()
        if not stripped_line:
            kept_lines.append("")
            continue

        if not _IMPORT_LINE_PATTERN.fullmatch(stripped_line):
            kept_lines.append(line)
            continue

        tokens = re.findall(r"[a-z_][a-z0-9_]*", stripped_line)
        filtered_tokens = [token for token in tokens if token not in {"import", "from", "as"}]
        import_tokens.extend(filtered_tokens)

    import_section = ""
    if import_tokens:
        import_section = "imports " + " ".join(import_tokens)

    return import_section, "\n".join(kept_lines)


def _normalize_code_identifiers(code: str) -> str:
    identifier_map: dict[str, str] = {}
    reserved_tokens = {
        "and", "as", "assert", "break", "class", "continue", "def", "del", "elif", "else", "except",
        "false", "finally", "for", "from", "if", "import", "in", "is", "lambda", "none", "nonlocal",
        "not", "or", "pass", "raise", "return", "self", "true", "try", "while", "with", "yield",
    }

    class_counter = 0
    func_counter = 0
    arg_counter = 0

    for match in _CLASS_DEF_PATTERN.finditer(code):
        name = match.group(1)
        if name not in identifier_map:
            class_counter += 1
            identifier_map[name] = f"CLASS_{class_counter}"

    for match in _FUNC_DEF_PATTERN.finditer(code):
        name = match.group(1)
        if name not in identifier_map:
            func_counter += 1
            identifier_map[name] = f"FUNC_{func_counter}"

    for match in _FUNC_ARG_PATTERN.finditer(code):
        args = match.group(1)
        for raw_arg in args.split(","):
            arg = raw_arg.strip()
            if not arg:
                continue
            arg_name = arg.split("=")[0].strip()
            arg_name = arg_name.lstrip("*")
            if not arg_name or arg_name in reserved_tokens or arg_name in identifier_map:
                continue
            arg_counter += 1
            identifier_map[arg_name] = f"ARG_{arg_counter}"

    def _replace_identifier(match: re.Match[str]) -> str:
        token = match.group(0)
        if token in identifier_map:
            return identifier_map[token]
        return token

    return _IDENTIFIER_PATTERN.sub(_replace_identifier, code)


def _normalize_code_segment_for_near_dedup(block: str) -> str:
    opening_fence, _, remainder = block.partition("\n")
    language = opening_fence[3:].strip()
    code = remainder[:-3] if remainder.endswith("```") else remainder

    # Step 5: normalize code fences and replace placeholder comments.
    code = _PLACEHOLDER_COMMENT_PATTERN.sub("MISSING_CODE", code)

    # Step 6: normalize identifiers such as class, function, and argument names.
    code = _normalize_code_identifiers(code)

    # Step 8: normalize imports into a compact token-based representation.
    imports_text, code_without_imports = _extract_import_tokens(code)

    # Step 9: drop low-value syntax punctuation from non-import code.
    code_without_imports = _PUNCTUATION_TO_DROP_PATTERN.sub(" ", code_without_imports)

    parts = []
    if language:
        parts.append(f"code {language}")
    else:
        parts.append("code")
    if imports_text:
        parts.append(imports_text)
    if code_without_imports.strip():
        parts.append(code_without_imports.strip())

    return "\n".join(parts)


def normalize_instruction_text_for_near_dedup(text: str) -> str:
    """
    Normalize an instruction for near-deduplication.

    Pipeline:
    1. Apply Unicode normalization.
    2. Remove leading and trailing whitespace.
    3. Lowercase the full instruction.
    4. Separate prose and fenced code blocks.
    5. Normalize code fences and map placeholder comments such as
       "# Your code here" and "TODO" to "MISSING_CODE".
    6. Normalize identifiers to abstract placeholders such as CLASS_1,
       FUNC_1, and ARG_1.
    7. Collapse repeated whitespace and blank lines.
    8. Normalize imports into a compact representation like
       "imports tensorflow tensorflow.keras.models sequential".
    9. Drop low-value syntax punctuation from non-import code:
       "(", ")", ":", ".", ",", "=".
    """
    if text is None:
        return ""

    # Step 1: apply Unicode normalization.
    normalized_text = unicodedata.normalize("NFKC", str(text))

    # Step 2: remove leading and trailing whitespace.
    normalized_text = normalized_text.strip()

    # Step 3: lowercase the full instruction.
    normalized_text = normalized_text.lower()
    normalized_text = normalized_text.replace("\r\n", "\n").replace("\r", "\n")
    normalized_text = _ZERO_WIDTH_CHARS.sub("", normalized_text)

    parts = []
    cursor = 0

    # Step 4: separate prose and fenced code blocks.
    for match in _CODE_BLOCK_PATTERN.finditer(normalized_text):
        prose = normalized_text[cursor:match.start()]
        if prose.strip():
            prose = prose.replace("`", "")
            prose = prose.replace("_", " ")
            parts.append(prose.strip())

        code_block = match.group(0)

        # Step 5: normalize code fences and placeholder comments.
        # Step 6: normalize identifiers.
        # Step 8: normalize imports.
        # Step 9: drop low-value syntax punctuation from non-import code.
        parts.append(_normalize_code_segment_for_near_dedup(code_block))
        cursor = match.end()

    tail = normalized_text[cursor:]
    if tail.strip():
        tail = tail.replace("`", "")
        tail = tail.replace("_", " ")
        parts.append(tail.strip())

    combined_text = "\n\n".join(parts)

    # Step 7: collapse repeated whitespace and blank lines.
    combined_text = _HORIZONTAL_WHITESPACE.sub(" ", combined_text)
    combined_text = re.sub(r" *\n *", "\n", combined_text)
    combined_text = re.sub(r"\n+", "\n", combined_text)

    return combined_text.strip()


def normalize_instruction_for_near_dedup(example: dict) -> dict:
    return {"near_dedup_normalized_text": normalize_instruction_text_for_near_dedup(example["instruction"])}


def shingle_normalized_text(normalized_text: str, shingle_size: int = 5) -> list[str]:
    """Create token shingles from normalized text for near-deduplication."""
    tokens = normalized_text.split()
    if not tokens:
        return []
    if shingle_size <= 1:
        return tokens
    if len(tokens) < shingle_size:
        return [" ".join(tokens)]
    return [" ".join(tokens[index:index + shingle_size]) for index in range(len(tokens) - shingle_size + 1)]


def _require_datasketch():
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError as exc:
        raise ImportError(
            "datasketch is required for near-duplicate detection. Install it with `pip install datasketch`."
        ) from exc
    return MinHash, MinHashLSH


def compute_minhash_signature(
    shingles: Iterable[str],
    num_perm: int = 128,
):
    """Compute a datasketch MinHash signature from a sample's shingles."""
    MinHash, _ = _require_datasketch()

    signature = MinHash(num_perm=num_perm)
    for shingle in sorted(set(shingles)):
        signature.update(shingle.encode("utf-8"))
    return signature


def compute_minhash_signature_for_sample(
    example: dict,
    text_field: str = "near_dedup_normalized_text",
    shingle_size: int = 5,
    num_perm: int = 128,
) -> dict:
    """Map-friendly wrapper that computes shingles and a MinHash signature for one sample."""
    normalized_text = example.get(text_field) or normalize_instruction_text_for_near_dedup(example["instruction"])
    shingles = shingle_normalized_text(normalized_text, shingle_size=shingle_size)
    signature = compute_minhash_signature(shingles, num_perm=num_perm)
    return {
        "near_dedup_shingles": shingles,
        "near_dedup_minhash_signature": signature.digest().tolist(),
    }


def build_lsh_index(
    signatures: list,
    threshold: float = 0.8,
    num_perm: int = 128,
):
    """Build a datasketch MinHashLSH index over MinHash signatures."""
    _, MinHashLSH = _require_datasketch()

    if not signatures:
        return MinHashLSH(threshold=threshold, num_perm=num_perm)

    index = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for sample_index, signature in enumerate(signatures):
        index.insert(sample_index, signature)
    return index


def score_candidate_pair_with_jaccard(
    left_shingles: Iterable[str],
    right_shingles: Iterable[str],
) -> float:
    """Score a candidate pair using exact Jaccard similarity over shingles."""
    left_set = set(left_shingles)
    right_set = set(right_shingles)

    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0

    intersection_size = len(left_set & right_set)
    union_size = len(left_set | right_set)
    return intersection_size / union_size


def _prepare_near_dedup_records(
    dataset: Dataset,
    shingle_size: int,
    num_perm: int,
) -> list[dict]:
    records = []
    for sample in dataset:
        normalized_text = normalize_instruction_text_for_near_dedup(sample["instruction"])
        shingles = shingle_normalized_text(normalized_text, shingle_size=shingle_size)
        minhash_signature = compute_minhash_signature(shingles, num_perm=num_perm)
        records.append(
            {
                "id": sample["id"],
                "source": sample["source"],
                "normalized_text": normalized_text,
                "shingles": shingles,
                "minhash_signature": minhash_signature,
            }
        )
    return records


def export_near_duplicates_with_minhash_lsh(
    train_ds_list,
    test_ds_list,
    threshold: float = 0.8,
    shingle_size: int = 5,
    num_perm: int = 128,
    lsh_threshold: float | None = None,
    output_path: str | Path = DEFAULT_NEAR_DUPLICATES_JSONL,
) -> Dataset:
    """
    Find near-duplicate train/test instruction pairs with MinHash and LSH from datasketch.

    Steps:
    1. Load the training and test datasets separately with `load_instructions()`.
    2. Normalize each sample with `normalize_instruction_text_for_near_dedup()`.
    3. Convert each normalized sample into shingles with `shingle_normalized_text()`.
    4. Compute a datasketch `MinHash` signature for each sample with `compute_minhash_signature()`.
    5. Build a datasketch `MinHashLSH` index over the test set only with `build_lsh_index()`.
    6. Query each training sample against the test-set LSH index to get candidate matches.
    7. Score each candidate pair with exact Jaccard similarity over shingles using
       `score_candidate_pair_with_jaccard()`.
    8. Write every pair whose Jaccard score meets or exceeds `threshold` to a JSONL file with:
       `train_source`, `train_id`, `test_source`, `test_id`, `jaccard`,
       `normalized_train_text`, and `normalized_test_text`.
    9. Print summary statistics and return the training dataset after removing train samples
       that matched at least one near-duplicate test sample.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if lsh_threshold is None:
        lsh_threshold = threshold

    # Step 1: load the training and test datasets separately.
    train_dataset = load_instructions(train_ds_list)
    test_dataset = load_instructions(test_ds_list)
    print(f"Loaded {len(train_dataset)} training samples and {len(test_dataset)} test samples.")

    # Step 2: normalize each sample for near-deduplication.
    # Step 3: shingle each normalized sample.
    # Step 4: compute a datasketch MinHash signature for each sample.
    train_records = _prepare_near_dedup_records(train_dataset, shingle_size=shingle_size, num_perm=num_perm)
    test_records = _prepare_near_dedup_records(test_dataset, shingle_size=shingle_size, num_perm=num_perm)
    print(f"Prepared MinHash records for {len(train_records)} training samples and {len(test_records)} test samples.")

    # Step 5: build an LSH index on the test set only.
    test_lsh_index = build_lsh_index(
        [record["minhash_signature"] for record in test_records],
        threshold=lsh_threshold,
        num_perm=num_perm,
    )
    print(f"Built LSH index with threshold {lsh_threshold}.")

    near_duplicate_rows = []
    matched_train_indices = set()

    # Step 6: query each train sample against the test LSH index.
    for train_index, train_record in enumerate(train_records):
        candidate_test_indices = test_lsh_index.query(train_record["minhash_signature"])

        for test_index in candidate_test_indices:
            test_record = test_records[test_index]

            # Step 7: score each candidate pair with exact Jaccard on shingles.
            jaccard_score = score_candidate_pair_with_jaccard(train_record["shingles"], test_record["shingles"])

            if jaccard_score < threshold:
                continue

            matched_train_indices.add(train_index)
            near_duplicate_rows.append(
                {
                    "train_source": train_record["source"],
                    "train_id": train_record["id"],
                    "test_source": test_record["source"],
                    "test_id": test_record["id"],
                    "jaccard": jaccard_score,
                    "normalized_train_text": train_record["normalized_text"],
                    "normalized_test_text": test_record["normalized_text"],
                }
            )

    near_duplicate_rows.sort(
        key=lambda row: (-row["jaccard"], row["train_source"], str(row["train_id"]), row["test_source"], str(row["test_id"]))
    )
    print(f"Found {len(near_duplicate_rows)} near-duplicate pairs.")

    # Step 8: write near-duplicate pairs to a JSONL file.
    with output_path.open("w", encoding="utf-8") as handle:
        for row in near_duplicate_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    filtered_train_indices = [index for index in range(len(train_dataset)) if index not in matched_train_indices]
    filtered_train_dataset = train_dataset.select(filtered_train_indices)

    # Step 9: print summary statistics.
    print(f"Saved near-duplicate report to {output_path}")
    print(f"Arguments: threshold={threshold}, lsh_threshold={lsh_threshold}, shingle_size={shingle_size}, num_perm={num_perm}")
    print(f"Raw training set size: {len(train_dataset)}")
    print(f"Raw test set size: {len(test_dataset)}")
    print(f"Number of near duplicate pairs: {len(near_duplicate_rows)}")
    print(f"Training set size after removal: {len(filtered_train_dataset)}")

    return filtered_train_dataset


def hash_normalized_instruction_text(normalized_text: str) -> str:
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def normalize_and_hash_instruction(example: dict) -> dict:
    normalized_text = normalize_instruction_text(example["instruction"])
    return {
        "normalized_text": normalized_text,
        "instruction_hash": hash_normalized_instruction_text(normalized_text),
    }


def export_exact_duplicates(
    dataset: Dataset,
    output_path: str | Path = DEFAULT_DUPLICATES_JSONL,
) -> Dataset:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = dataset.map(normalize_and_hash_instruction)

    hash_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(dataset):
        hash_to_indices[item["instruction_hash"]].append(index)

    canonical_indices = []
    duplicate_rows = []

    for indices in hash_to_indices.values():
        canonical_index = indices[0]
        canonical_indices.append(canonical_index)
        canonical_item = dataset[canonical_index]

        if len(indices) == 1:
            continue

        for duplicate_index in indices:
            duplicate_item = dataset[duplicate_index]
            duplicate_rows.append(
                {
                    "duplicate_id": duplicate_item["id"],
                    "canonical_id": canonical_item["id"],
                    "source": duplicate_item["source"],
                    "hash": duplicate_item["instruction_hash"],
                    "normalized_text": duplicate_item["normalized_text"],
                }
            )

    duplicate_rows.sort(key=lambda row: (row["hash"], str(row["duplicate_id"])))
    with output_path.open("w", encoding="utf-8") as handle:
        for row in duplicate_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    deduplicated_dataset = dataset.select(sorted(canonical_indices))

    duplicate_clusters = sum(1 for indices in hash_to_indices.values() if len(indices) > 1)
    duplicated_samples = len(duplicate_rows)
    total_samples = len(dataset)
    remaining_samples = len(deduplicated_dataset)

    print(f"Saved exact duplicate report to {output_path}")
    print(f"Number of samples processed: {total_samples}")
    print(f"Number of samples remaining: {remaining_samples}")
    print(f"Number of duplicated samples recorded: {duplicated_samples}")
    print(f"Number of duplicate clusters: {duplicate_clusters}")
    print(f"Number of unique hashes: {len(hash_to_indices)}")

    return deduplicated_dataset


def _load_json_records(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find duplicate file: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            return data
        raise ValueError(f"Expected a JSON array in {path}")

    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _clear_hub_dataset(repo_id: str) -> None:
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception:
        return

    for repo_file in repo_files:
        if repo_file == ".gitattributes":
            continue
        api.delete_file(
            path_in_repo=repo_file,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Remove stale dataset file: {repo_file}",
        )


def _extract_solution_code_with_fences(text) -> str | None:
    if text is None:
        return None

    text = str(text)
    code_blocks = []
    for match in _CODE_BLOCK_PATTERN.finditer(text):
        block = match.group(0).strip()
        opening_fence, _, remainder = block.partition("\n")
        code = remainder[:-3] if remainder.endswith("```") else remainder
        code_blocks.append(code.strip())
    if code_blocks:
        return "\n\n".join(code_blocks)

    stripped_text = text.strip()
    return stripped_text or None


def _standardize_split_dataset(source: str, dataset: Dataset) -> Dataset:
    index_column = INDEX_COLUMNS[source]
    language_column = LANGUAGE_COLUMNS[source]
    instruction_column = INSTRUCTION_COLUMNS[source]
    solution_column = SOLUTION_COLUMNS[source]
    test_column = TEST_COLUMNS[source]

    standardized_rows = []
    for row in dataset:
        raw_language = row.get(language_column)
        if source == "McEval" and raw_language is not None:
            raw_language = str(raw_language).split("/", 1)[0]

        language = LANGUAGE_MAPPINGS.get(raw_language)
        if language not in LANGUAGES:
            continue

        standardized_rows.append(
            {
                "index": None if row.get(index_column) is None else str(row.get(index_column)),
                "language": language,
                "instruction": row.get(instruction_column),
                "solution": _extract_solution_code_with_fences(row.get(solution_column)) if solution_column else None,
                "test": row.get(test_column) if test_column else None,
            }
        )

    if not standardized_rows:
        return Dataset.from_dict(
            {
                "index": [],
                "language": [],
                "instruction": [],
                "solution": [],
                "test": [],
            },
            features=STANDARD_FEATURES,
        )

    return Dataset.from_list(standardized_rows, features=STANDARD_FEATURES)


def remove_duplicates(
    exact_duplicates_path: str | Path,
    near_duplicates_path: str | Path,
    train_ds_list: list[str],
    test_ds_list: list[str],
    repo_id: str | None = None,
) -> DatasetDict:
    requested_sources = list(dict.fromkeys(train_ds_list + test_ds_list))
    all_datasets = load_instruction_datasets(requested_sources)
    unknown_sources = [source for source in requested_sources if source not in all_datasets]
    if unknown_sources:
        raise ValueError(f"Unsupported dataset sources: {unknown_sources}")

    train_datasets = {source: all_datasets[source] for source in train_ds_list}
    test_datasets = {source: all_datasets[source] for source in test_ds_list}

    train_sources = set(train_ds_list)
    test_sources = set(test_ds_list)

    keep_indices_by_source = {
        source: set(range(len(dataset)))
        for source, dataset in all_datasets.items()
    }
    id_to_index_by_source: dict[str, dict[str, int]] = {}
    for source, dataset in all_datasets.items():
        id_field = INDEX_COLUMNS[source]
        id_to_index_by_source[source] = {
            str(sample[id_field]): index
            for index, sample in enumerate(dataset)
        }

    exact_clusters: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in _load_json_records(exact_duplicates_path):
        canonical_id = row.get("canonical_id")
        duplicate_id = row.get("duplicate_id")
        source = row.get("source")
        if canonical_id is None or duplicate_id is None or source is None:
            continue

        exact_clusters[str(canonical_id)].add((str(source), str(duplicate_id)))

    for cluster in exact_clusters.values():
        existing_records = [
            (source, sample_id)
            for source, sample_id in cluster
            if sample_id in id_to_index_by_source.get(source, {})
        ]
        if not existing_records:
            continue

        keep_source, keep_sample_id = min(
            existing_records,
            key=lambda item: (0 if item[0] in test_sources else 1, item[0], item[1]),
        )
        for source, sample_id in existing_records:
            if (source, sample_id) == (keep_source, keep_sample_id):
                continue
            keep_indices_by_source[source].discard(id_to_index_by_source[source][sample_id])

    for row in _load_json_records(near_duplicates_path):
        train_source = row.get("train_source")
        train_id = row.get("train_id")
        if train_source is None or train_id is None:
            continue

        train_sample_id = str(train_id)
        if train_source not in train_sources:
            continue
        if train_sample_id not in id_to_index_by_source.get(train_source, {}):
            continue

        keep_indices_by_source[train_source].discard(id_to_index_by_source[train_source][train_sample_id])

    deduped_splits = {}
    for source in train_ds_list:
        selected_indices = sorted(keep_indices_by_source[source])
        deduped_splits[f"train_{source.replace('-', '_')}"] = _standardize_split_dataset(
            source,
            train_datasets[source].select(selected_indices),
        )
    for source in test_ds_list:
        selected_indices = sorted(keep_indices_by_source[source])
        deduped_splits[f"test_{source.replace('-', '_')}"] = _standardize_split_dataset(
            source,
            test_datasets[source].select(selected_indices),
        )

    deduped_dataset_dict = DatasetDict(deduped_splits)
    if repo_id:
        _clear_hub_dataset(repo_id)
        deduped_dataset_dict.push_to_hub(repo_id)
    return deduped_dataset_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize and deduplicate instruction datasets.")
    parser.add_argument(
        "--mode",
        choices=["exact", "near", "remove"],
        default="exact",
        help="Deduplication mode to run.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["OSS-Instruct", "McEval-Instruct", "McEval"],
        help="Instruction datasets to load for exact deduplication.",
    )
    parser.add_argument(
        "--duplicates-output",
        default=str(DEFAULT_DUPLICATES_JSONL),
        help="Path to the JSONL file containing exact duplicates.",
    )
    parser.add_argument(
        "--train-datasets",
        nargs="+",
        default=["OSS-Instruct", "McEval-Instruct"],
        help="Instruction datasets to load for the training split in near deduplication.",
    )
    parser.add_argument(
        "--test-datasets",
        nargs="+",
        default=["McEval"],
        help="Instruction datasets to load for the test split in near deduplication.",
    )
    parser.add_argument(
        "--near-duplicates-output",
        default=str(DEFAULT_NEAR_DUPLICATES_JSONL),
        help="Path to the JSONL file containing near-duplicate train/test pairs.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Optional Hugging Face dataset repo id to upload the filtered splits to.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Exact Jaccard threshold used to keep near-duplicate pairs.",
    )
    parser.add_argument(
        "--lsh-threshold",
        type=float,
        default=None,
        help="Optional LSH query threshold. Defaults to the exact Jaccard threshold.",
    )
    parser.add_argument(
        "--shingle-size",
        type=int,
        default=5,
        help="Token shingle size used for near deduplication.",
    )
    parser.add_argument(
        "--num-perm",
        type=int,
        default=128,
        help="Number of permutations used for datasketch MinHash.",
    )
    args = parser.parse_args()

    if args.mode == "near":
        if not args.train_datasets or not args.test_datasets:
            parser.error("--train-datasets and --test-datasets are required when --mode near is used.")

        export_near_duplicates_with_minhash_lsh(
            train_ds_list=args.train_datasets,
            test_ds_list=args.test_datasets,
            threshold=args.threshold,
            shingle_size=args.shingle_size,
            num_perm=args.num_perm,
            lsh_threshold=args.lsh_threshold,
            output_path=args.near_duplicates_output,
        )
        return

    if args.mode == "remove":
        if not args.train_datasets or not args.test_datasets:
            parser.error("--train-datasets and --test-datasets are required when --mode remove is used.")

        remove_duplicates(
            exact_duplicates_path=args.duplicates_output,
            near_duplicates_path=args.near_duplicates_output,
            train_ds_list=args.train_datasets,
            test_ds_list=args.test_datasets,
            repo_id=args.repo_id,
        )
        return

    dataset = load_instructions(args.datasets)
    export_exact_duplicates(dataset, args.duplicates_output)


if __name__ == "__main__":
    main()
