"""Task registry with the no-replay guard (Invariant I4, test T9). Reuses the thesis dataset sources and template."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Mirrors utils/data/data_collator.py::DataCollator.decoder_call, which builds the generation prompt.
PROMPT_TEMPLATE = "input: {instruction}\noutput: "


def format_prompt(instruction: str) -> str:
    return PROMPT_TEMPLATE.format(instruction=instruction)


@dataclass(frozen=True)
class Example:
    prompt: str
    answer: str


@dataclass
class TaskData:
    """Training data of the current task only: D_k^tr and the held-out slice H_k."""
    index: int
    name: str
    train: List[Example]
    heldout: List[Example]


class ReplayError(RuntimeError):
    pass


# A source returns the examples of one split ("train" or "test") of one task.
TaskSource = Callable[[str, str], List[Example]]


def split_heldout(examples: Sequence[Example], frac: float, max_n: int, seed: int) -> Tuple[List[Example], List[Example]]:
    idx = list(range(len(examples)))
    random.Random(seed).shuffle(idx)
    n_h = min(max_n, int(round(frac * len(examples))))
    held = set(idx[:n_h])
    return ([examples[i] for i in range(len(examples)) if i not in held], [examples[i] for i in idx[:n_h]])


class TaskRegistry:
    def __init__(self, task_names: Sequence[str], source: TaskSource, heldout_frac: float, heldout_max: int, seed: int):
        self.task_names = list(task_names)
        self._source = source
        self._frac, self._max, self._seed = heldout_frac, heldout_max, seed
        self._current: Optional[int] = None
        self._finished: set = set()
        self._tests: Dict[int, Tuple[Example, ...]] = {}

    def begin_task(self, k: int) -> None:
        if k in self._finished or (self._current is not None and k <= self._current):
            raise ReplayError(f"task {k} was already learned")
        self._current = k

    def training_data(self, k: int) -> TaskData:
        if k != self._current or k in self._finished:
            raise ReplayError(f"refusing to load training data of task {k} while training task {self._current}")
        train = self._source(self.task_names[k], "train")
        tr, held = split_heldout(train, self._frac, self._max, self._seed + k)
        return TaskData(index=k, name=self.task_names[k], train=tr, heldout=held)

    def end_task(self, k: int) -> None:
        self._finished.add(k)

    def resume(self, n_done: int) -> None:
        """Mark tasks 0..n_done-1 as already learned (restarting from a checkpoint)."""
        self._finished = set(range(n_done))
        self._current = n_done - 1 if n_done > 0 else None

    def evaluation_view(self) -> "EvalView":
        return EvalView(self)


class EvalView:
    """Read-only access to test splits of tasks seen so far; for evaluation code only."""

    def __init__(self, registry: TaskRegistry):
        self._r = registry

    def test_split(self, j: int) -> Tuple[Example, ...]:
        if self._r._current is None or j > self._r._current:
            raise ReplayError(f"test split of task {j} is not available yet")
        if j not in self._r._tests:
            self._r._tests[j] = tuple(self._r._source(self._r.task_names[j], "test"))
        return self._r._tests[j]


# ---- thesis dataset sources ---------------------------------------------------

def _cap(ds, n: int, seed: int):
    ds = ds.shuffle(seed=seed)
    return ds.select(range(min(n, len(ds)))) if n >= 0 else ds


def hf_source(benchmark: str, seed: int, max_train: int, max_test: int) -> TaskSource:
    """Same repos, filters and columns as utils/data/data_utils.py (create_codetask_dataset / executable loaders).
    Subsets are drawn with `seed` (cfg.data_seed), so every method and training seed sees the same examples."""

    def load(task: str, split: str) -> List[Example]:
        from datasets import concatenate_datasets, load_dataset
        n = max_train if split == "train" else max_test
        if benchmark == "codetask":
            ds = load_dataset("dongg18/CODETASK_with_instruction_pool",
                              data_files={split: f"{task}/{split}-*.parquet"}, split=split)
            return [Example(str(r["input"]), str(r["output"])) for r in _cap(ds, n, seed)]
        repo = "ankhanhtran02/CL4Code-executable-datasets"
        if split == "train":
            parts = [load_dataset(repo, split=s).filter(lambda r: r["language"] == task and r["solution"] is not None)
                     for s in ("train_OSS_Instruct", "train_McEval_Instruct")]
            ds = _cap(concatenate_datasets(parts), n, seed)
        else:
            ds = load_dataset(repo, split="test_McEval").filter(lambda r: r["language"] == task and r["test"] is not None)
            ds = ds.select(range(min(n, len(ds))))
        return [Example(r["instruction"], r["solution"]) for r in ds]

    return load
