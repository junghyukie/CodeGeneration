"""Reclassify ise-uiuc/Magicoder-OSS-Instruct-75K samples whose `problem` text
never names its own tagged language, then push the language-specific subset to
the `train_OSS_Instruct` split of ankhanhtran02/CL4Code-executable-datasets.

Pipeline, run per sample against its OWN tagged language:
  Pass 1 (name alias)      - does `problem` contain any spelling/casing of the
                              language's own name (python/Python/PY, c++/C++/cpp, ...)?
  Pass 2 (framework/syntax) - does it name a framework/library/stdlib call typical
                              of that language, or contain a language-specific
                              syntax fragment (regex over the raw text)?
  Pass 3 (code-fence lexer) - does a Pygments lexer's analyse_text() score the
                              language highest among the 9 candidates, when run
                              over the ```-fenced code blocks in the problem?

A sample is "language-specific" if it clears any of the three passes for its
own tag; otherwise it's "still generic" and excluded from the push.

For `python` specifically, the language-specific pool is randomly subsampled
down to `--sample-n` (default 4000, seed `--seed` default 42) before pushing,
since it vastly outnumbers the other 8 languages. Every other language pushes
its full language-specific pool as-is.

Requires: `datasets` (already pinned in requirements.txt) and `pygments`
(not currently pinned - add `pygments` to requirements.txt / your venv before
running this script).

Usage:
    # Dry run (default): writes the combined push set to --output-dir and
    # prints per-language statistics. Does NOT touch the Hugging Face Hub.
    python executable_dataset/reclassify_oss_instruct.py \
        --output-dir executable_dataset/output/oss_instruct_reclassification

    # After reviewing the dry-run stats, push for real:
    python executable_dataset/reclassify_oss_instruct.py --push

    # offline / repeated runs against an already-downloaded parquet snapshot:
    python executable_dataset/reclassify_oss_instruct.py --parquet-path /path/to/train.parquet
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable

from pygments.lexers import get_lexer_by_name

SOURCE_DATASET_NAME = "ise-uiuc/Magicoder-OSS-Instruct-75K"
TARGET_REPO_ID = "ankhanhtran02/CL4Code-executable-datasets"
TARGET_SPLIT = "train_OSS_Instruct"
TARGET_CONFIG = "default"

# Canonical language keys match LANGUAGE_MAPPINGS' values in preprocess.py, and
# the `language` column values already used in the target dataset (verified
# against ankhanhtran02/CL4Code-executable-datasets:train_OSS_Instruct).
LANGUAGES = ["python", "cpp", "swift", "rust", "csharp", "java", "php", "typescript", "shell"]

LANG_MAP = {  # dataset's raw `lang` column -> our canonical key
    "python": "python", "cpp": "cpp", "swift": "swift", "rust": "rust",
    "csharp": "csharp", "java": "java", "php": "php",
    "typescript": "typescript", "shell": "shell",
}

# ---------------------------------------------------------------------------
# Pass 1: name aliases (case-sensitive, alnum-boundary matched)
# ---------------------------------------------------------------------------
ALIASES = {
    "python": ["python", "Python", "PYTHON", "py", "Py", "PY"],
    "cpp": ["c++", "C++", "cpp", "CPP", "Cpp", "cplusplus", "CPlusPlus",
            "c plus plus", "C Plus Plus"],
    "swift": ["swift", "Swift", "SWIFT"],
    "rust": ["rust", "Rust", "RUST"],
    "csharp": ["c#", "C#", "csharp", "CSharp", "CSHARP", "c-sharp", "C-Sharp",
               "c sharp", "C Sharp"],
    "java": ["java", "Java", "JAVA"],
    "php": ["php", "PHP", "Php"],
    "typescript": ["typescript", "TypeScript", "TYPESCRIPT", "ts", "TS"],
    "shell": ["shell", "Shell", "SHELL", "bash", "Bash", "BASH", "sh", "Sh", "SH"],
}

# ---------------------------------------------------------------------------
# Pass 2: framework / library / stdlib keyword lexicons + syntax fingerprints
# ---------------------------------------------------------------------------
FRAMEWORK_KEYWORDS = {
    "python": ["Django", "Flask", "NumPy", "numpy", "Pandas", "pandas", "PyTorch",
               "TensorFlow", "Keras", "scikit-learn", "sklearn", "Matplotlib",
               "FastAPI", "SQLAlchemy", "Celery", "pytest", "Jupyter", "Anaconda",
               "wxPython", "Tkinter", "tkinter", "BeautifulSoup", "OpenCV", "cv2",
               "DRF", "PyPI", "pip install", "conda install", "PySide", "PyQt"],
    "cpp": ["STL", "Boost", "CMake", "Qt", "libstdc++", "iostream", "cstdio",
            "cstdlib", "cout", "cin", "cerr", "nullptr", "std::vector",
            "std::string", "std::map", "unordered_map", "malloc", "makefile",
            "Makefile"],
    "swift": ["UIKit", "SwiftUI", "Xcode", "CocoaPods", "Foundation", "Combine",
              "Core Data", "CoreData", "ARKit", "SpriteKit", "Alamofire",
              "NSObject", "UIViewController", "@testable", "Swift Package Manager"],
    "rust": ["Cargo", "cargo build", "crates.io", "tokio", "serde", "actix",
             "Rocket", "wasm-bindgen", "rustc", "impl ", "println!", "unwrap()",
             "Option<", "Result<", "match ", "let mut", "trait ", "derive("],
    "csharp": [".NET", "ASP.NET", "LINQ", "NuGet", "Xamarin", "Entity Framework",
               "WPF", "WinForms", "Console.WriteLine", "using System", "namespace ",
               "async Task", "IEnumerable", "IDisposable"],
    "java": ["Spring", "Maven", "Gradle", "Android", "JVM", "Hibernate",
             "Eclipse Paho", "JUnit", "Servlet", "JDBC", "Jakarta", "Apache Kafka",
             "System.out.println", "public static void main", "extends ",
             "implements ", "import java.", "@Override", "ArrayList<"],
    "php": ["Laravel", "Symfony", "Composer", "WordPress", "CodeIgniter",
            "CakePHP", "Eloquent", "Blade", "<?php", "$this->", "$_GET",
            "$_POST", "$_SESSION", "namespace App", "use Illuminate"],
    "typescript": ["React", "Angular", "Vue", "Express", "Node.js", "NestJS",
                   "Redux", "Webpack", "npm install", "Material-UI", "Jest",
                   "RxJS", "JavaScript", "javascript", "interface ", "=> {",
                   "export default", "import {", "Promise<", "async function",
                   "React.", "useState", "useEffect"],
    "shell": ["crontab", "systemd", "chmod", "chown", "xargs", "/bin/bash",
              "/bin/sh", "#!/bin/", "awk ", "sed ", "grep ", "Unix-based",
              "command line", "terminal", "shell script", "Bash script",
              "environment variable"],
}

SYNTAX_PATTERNS = {
    "python": [r"\bdef \w+\(", r"\bself\.", r"\b__init__\b", r"\belif\b",
               r"\bexcept \w+:", r"f['\"].*\{.*\}['\"]", r"\blambda\b"],
    "cpp": [r"#include\s*<", r"\bstd::", r"\bcout\s*<<", r"\bcin\s*>>",
            r"\btemplate\s*<", r"\bnamespace\s+\w+", r"int main\s*\("],
    "swift": [r"\bfunc \w+\(", r"\bguard let\b", r"\boverride func\b",
              r"\bvar \w+:\s*\w+", r"\blet \w+:\s*\w+", r"->\s*Bool\b",
              r"\bprotocol \w+"],
    "rust": [r"\bfn \w+\(", r"\blet mut\b", r"\bimpl\s+\w+", r"\bmatch \w+\s*\{",
             r"println!\(", r"\bpub fn\b", r"&mut\s+\w+"],
    "csharp": [r"\busing System", r"\bpublic class \w+", r"\bvoid Main\s*\(",
               r"Console\.WriteLine", r"=>\s*\w", r"\{\s*get;\s*set;\s*\}"],
    "java": [r"\bpublic class \w+", r"public static void main",
             r"System\.out\.println", r"\bnew ArrayList<", r"@Override"],
    "php": [r"<\?php", r"\$this->\w+", r"\$_GET\[", r"\$_POST\[",
            r"\bfunction \w+\(.*\)\s*\{"],
    "typescript": [r"\binterface \w+", r":\s*(string|number|boolean)\b",
                    r"\bconst \w+\s*=", r"\bexport default\b", r"\bimport \{",
                    r"Promise<", r"=>\s*\{"],
    "shell": [r"#!/bin/(ba)?sh", r"\$\{?\d\}?", r"\becho\s+", r"\bif \[",
              r"\bfi\b", r"\bdone\b", r"\$\(.*\)"],
}

# ---------------------------------------------------------------------------
# Pass 3: Pygments lexer scoring of ```-fenced code blocks
# ---------------------------------------------------------------------------
CANDIDATE_LEXER_NAMES = {
    "python": ["python"], "cpp": ["cpp"], "swift": ["swift"], "rust": ["rust"],
    "csharp": ["csharp"], "java": ["java"], "php": ["php"],
    "typescript": ["typescript", "javascript"],  # dataset tags plain JS as "typescript"
    "shell": ["bash"],
}

FENCE_RE = re.compile(r"```[ \t]*\w*[ \t]*\n?(.*?)```", re.DOTALL)


def build_pattern(alias: str) -> re.Pattern:
    """Alnum-boundary match, but only on sides whose edge char is alnum.

    "Django" needs a boundary on both sides; "impl " / "$this->" / "<?php"
    already end or start on space/punctuation, so adding one there would
    incorrectly reject real matches (e.g. the trailing-space case would
    require the *next* character after the space to also be non-alnum).
    """
    prefix = r"(?<![A-Za-z0-9])" if alias[0].isalnum() else ""
    suffix = r"(?![A-Za-z0-9])" if alias[-1].isalnum() else ""
    return re.compile(prefix + re.escape(alias) + suffix)


COMPILED_ALIASES = {lang: [build_pattern(a) for a in aliases] for lang, aliases in ALIASES.items()}
COMPILED_FRAMEWORKS = {lang: [build_pattern(a) for a in kws] for lang, kws in FRAMEWORK_KEYWORDS.items()}
COMPILED_SYNTAX = {lang: [re.compile(p) for p in pats] for lang, pats in SYNTAX_PATTERNS.items()}
CANDIDATE_LEXERS = {lang: [get_lexer_by_name(n) for n in names] for lang, names in CANDIDATE_LEXER_NAMES.items()}


def has_alias(lang: str, text: str) -> bool:
    return any(p.search(text) for p in COMPILED_ALIASES[lang])


def matched_signals(lang: str, text: str) -> list[tuple[str, str]]:
    """Pass-2 hits: [(kind, matched keyword/pattern), ...]."""
    hits = []
    for p, raw in zip(COMPILED_FRAMEWORKS[lang], FRAMEWORK_KEYWORDS[lang]):
        if p.search(text):
            hits.append(("framework", raw))
    for p, raw in zip(COMPILED_SYNTAX[lang], SYNTAX_PATTERNS[lang]):
        if p.search(text):
            hits.append(("syntax", raw))
    return hits


def extract_code_blocks(text: str) -> str:
    blocks = FENCE_RE.findall(text)
    return "\n".join(b for b in blocks if b.strip())


def pygments_scores(code: str) -> dict[str, float]:
    scores = {}
    for lang, lexers in CANDIDATE_LEXERS.items():
        best = 0.0
        for lex in lexers:
            try:
                s = lex.analyse_text(code)
            except Exception:
                s = 0.0
            if s and s > best:
                best = s
        scores[lang] = best
    return scores


def classify_sample(orig_lang: str, text: str) -> dict:
    """Run passes 1-3 for one sample against its OWN tagged language only.
    Returns {"stage": "name_alias" | "framework_or_syntax" | "code_fence_lexer" | "still_generic", ...}.
    """
    if has_alias(orig_lang, text):
        return {"stage": "name_alias"}

    hits = matched_signals(orig_lang, text)
    if hits:
        return {"stage": "framework_or_syntax", "evidence": hits}

    code = extract_code_blocks(text)
    if code.strip():
        pg_scores = pygments_scores(code)
        own_pg = pg_scores.get(orig_lang, 0.0)
        own_argmax = max(pg_scores, key=pg_scores.get)
        if own_pg > 0.0 and own_argmax == orig_lang:
            return {"stage": "code_fence_lexer", "pygments_scores": pg_scores}

    return {"stage": "still_generic"}


def load_samples(parquet_path: str | None) -> Iterable[dict]:
    """Yields {"index", "lang", "problem", "solution"} dicts from the source dataset."""
    if parquet_path:
        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path, columns=["index", "lang", "problem", "solution"])
        cols = {name: table.column(name).to_pylist() for name in ("index", "lang", "problem", "solution")}
        for i in range(len(cols["index"])):
            yield {k: cols[k][i] for k in cols}
    else:
        from datasets import load_dataset
        ds = load_dataset(SOURCE_DATASET_NAME, split="train")
        for row in ds:
            yield {"index": row["index"], "lang": row["lang"], "problem": row["problem"], "solution": row["solution"]}


def collect_language_specific_samples(parquet_path: str | None) -> tuple[dict[str, list[dict]], dict[str, dict[str, int]]]:
    """Classifies every sample and buckets the language-specific ones (any of
    passes 1-3 hit) per language. Returns (records_by_lang, stage_counts).
    """
    records_by_lang: dict[str, list[dict]] = {lang: [] for lang in LANGUAGES}
    stage_counts = {lang: {"name_alias": 0, "framework_or_syntax": 0, "code_fence_lexer": 0, "still_generic": 0}
                    for lang in LANGUAGES}

    for row in load_samples(parquet_path):
        lang = LANG_MAP.get(row["lang"])
        if lang is None:
            continue
        text = row["problem"] or ""
        result = classify_sample(lang, text)
        stage_counts[lang][result["stage"]] += 1
        if result["stage"] != "still_generic":
            records_by_lang[lang].append(row)

    return records_by_lang, stage_counts


def sample_language_examples(records_by_lang: dict[str, list[dict]], lang: str = "python",
                              n: int = 4000, seed: int = 42) -> list[dict]:
    """Randomly sample n examples (default 4000) from `lang`'s language-specific
    pool (default python), with a fixed seed for reproducibility. Returns the
    full pool unchanged if it already has <= n examples."""
    pool = records_by_lang[lang]
    if len(pool) <= n:
        return list(pool)
    return random.Random(seed).sample(pool, n)


def map_to_target_schema(row: dict, lang: str) -> dict:
    """OSS-Instruct row -> ankhanhtran02/CL4Code-executable-datasets row.
    index->index, lang->language, problem->instruction, solution->solution;
    everything else (test/prompt/entry_point/signature) is null.
    """
    return {
        "index": str(row["index"]),
        "language": lang,
        "instruction": row["problem"],
        "solution": row["solution"],
        "test": None,
        "prompt": None,
        "entry_point": None,
        "signature": None,
    }


def remove_and_replace_language(repo_id: str, config_name: str, split: str, lang: str,
                                 new_records: list[dict], token: str | None = None) -> None:
    """Removes every existing row of `lang` from `split`, appends `new_records`
    (already mapped to the target schema) in its place, and pushes the updated
    split back to the Hub. Rows belonging to other languages are left untouched.
    """
    from datasets import Dataset, concatenate_datasets, load_dataset

    current = load_dataset(repo_id, config_name, split=split, token=token)
    kept = current.filter(lambda r: r["language"] != lang)
    new_ds = Dataset.from_list(new_records, features=current.features)
    updated = concatenate_datasets([kept, new_ds]) if len(kept) else new_ds
    updated.push_to_hub(repo_id, config_name=config_name, split=split, token=token)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet-path", default=None,
                         help="Local parquet snapshot of the source train split (skips HF `datasets` download).")
    parser.add_argument("--output-dir", default="executable_dataset/output/oss_instruct_reclassification")
    parser.add_argument("--sample-lang", default="python", help="Language to subsample (default: python).")
    parser.add_argument("--sample-n", type=int, default=4000, help="Sample size for --sample-lang (default: 4000).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42).")
    parser.add_argument("--repo-id", default=TARGET_REPO_ID)
    parser.add_argument("--split", default=TARGET_SPLIT)
    parser.add_argument("--push", action="store_true",
                         help="Actually remove+push to the Hub. Without this flag, only a dry run is performed "
                              "(writes the combined push set locally and prints statistics).")
    parser.add_argument("--hf-token", default=None, help="Optional explicit HF token (defaults to the cached login).")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records_by_lang, stage_counts = collect_language_specific_samples(args.parquet_path)

    print(f"{'Language':12s} {'alias':>7s} {'pass2':>7s} {'pass3':>7s} {'generic':>9s} {'lang_specific':>14s}")
    for lang in LANGUAGES:
        c = stage_counts[lang]
        specific = c["name_alias"] + c["framework_or_syntax"] + c["code_fence_lexer"]
        print(f"{lang:12s} {c['name_alias']:7d} {c['framework_or_syntax']:7d} "
              f"{c['code_fence_lexer']:7d} {c['still_generic']:9d} {specific:14d}")

    push_records_by_lang: dict[str, list[dict]] = {}
    for lang in LANGUAGES:
        pool = (sample_language_examples(records_by_lang, lang=args.sample_lang, n=args.sample_n, seed=args.seed)
                if lang == args.sample_lang else records_by_lang[lang])
        push_records_by_lang[lang] = [map_to_target_schema(r, lang) for r in pool]

    combined = [r for lang in LANGUAGES for r in push_records_by_lang[lang]]

    print(f"\nFinal push set (after {args.sample_lang!r} subsampled to n={args.sample_n}, seed={args.seed}):\n")
    print(f"{'Language':12s} {'count':>7s}")
    for lang in LANGUAGES:
        print(f"{lang:12s} {len(push_records_by_lang[lang]):7d}")
    print(f"{'TOTAL':12s} {len(combined):7d}")

    dry_run_path = out_dir / "oss_instruct_push_dryrun.jsonl"
    with open(dry_run_path, "w") as f:
        for r in combined:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote combined push set ({len(combined)} rows) to {dry_run_path}")

    if not args.push:
        print("\nDry run only - nothing was pushed. Re-run with --push once you've reviewed the stats/file above.")
        return

    print(f"\nPushing to {args.repo_id}:{args.split} (config={TARGET_CONFIG})...")
    for lang in LANGUAGES:
        print(f"  removing+replacing '{lang}' ({len(push_records_by_lang[lang])} new rows)...")
        remove_and_replace_language(args.repo_id, TARGET_CONFIG, args.split, lang,
                                     push_records_by_lang[lang], token=args.hf_token)
    print("Done.")


if __name__ == "__main__":
    main()
