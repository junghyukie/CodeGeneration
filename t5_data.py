from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from torch.utils.data import DataLoader
from datasets import load_dataset


class T5Dataset:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.task_list = [
            "CodeTrans", "CodeSearchNet", "BFP", "CONCODE",
            "TheVault_Csharp", "KodCode", "RunBugRun"
        ]
        self.text_key = {
            "CONCODE": "nl",
            "CodeTrans": "java",
            "CodeSearchNet": "code",
            "BFP": "buggy",
            "TheVault_Csharp": "code",
            "KodCode": "question",
            "RunBugRun": "buggy_code",
        }
        self.label_key = {
            "CONCODE": "code",
            "CodeTrans": "cs",
            "CodeSearchNet": "docstring",
            "BFP": "fixed",
            "TheVault_Csharp": "docstring",
            "KodCode": "solution",
            "RunBugRun": "fixed_code",
        }

        self.task_instructions = {
            "CONCODE": "Generate Java code from the following English description: ",
            "CodeTrans": "Translate the following Java code into C#: ",
            "CodeSearchNet": "Summarize the following ruby code into clear, concise English. Return only one sentence (no code, no quotes): ",
            "BFP": "Refactor or improve the following Java code: ",
            "TheVault_Csharp": "Summarize the following C# code into English: ",
            "KodCode": "Generate Python code from the following description: ",
            "RunBugRun": "Refactor or improve the following Ruby code: ",
        }

        self.max_input_length = {
            "CodeTrans": 320,
            "CodeSearchNet": 256,
            "BFP": 130,
            "CONCODE": 320,
            "TheVault_Csharp": 256,
            "KodCode": 256,
            "RunBugRun": 256,
        }
        self.max_target_length = {
            "CodeTrans": 256,
            "CodeSearchNet": 128,
            "BFP": 120,
            "CONCODE": 150,
            "TheVault_Csharp": 128,
            "KodCode": 256,
            "RunBugRun": 256,
        }

        # Datasets that don't provide official val/test splits (we split from train)
        self.train_only_tasks = {
            "KodCode": {"val": 1000, "test": 1000},
            "RunBugRun": {"val": 972, "test": 1000},
        }

    @staticmethod
    def _extract_first_paragraph(docstring: Any) -> str:
        if docstring is None:
            return ""
        if isinstance(docstring, (list, tuple)):
            s = " ".join(str(t) for t in docstring)
        else:
            s = str(docstring)
        s = s.replace("\n", "").replace("\r", "")
        s = " ".join(s.strip().split())
        return s

    def _split_train_only(self, dataset, task: str, split: str, split_seed: int = 42):
        sizes = self.train_only_tasks[task]
        test_size = sizes["test"]
        val_size = sizes["val"]

        tmp = dataset.train_test_split(test_size=test_size, seed=split_seed)
        test_ds = tmp["test"]

        tmp2 = tmp["train"].train_test_split(test_size=val_size, seed=split_seed)
        train_ds = tmp2["train"]
        val_ds = tmp2["test"]

        mapping = {"train": train_ds, "validation": val_ds, "test": test_ds}
        if split not in mapping:
            raise ValueError(f"Unknown split '{split}' for train-only task '{task}'")
        return mapping[split]

    def select_subset_ds(self, ds, k: int = 2000, seed: int = 0):
        np.random.seed(seed)
        num_samples = min(k, ds.shape[0])
        idx_total = np.random.choice(np.arange(ds.shape[0]), num_samples, replace=False)
        return ds.select(idx_total)

    def _preprocess_batch(self, examples, task: str, max_length: int = 512, max_target_length: int = 128):
        if task not in self.task_list:
            raise ValueError(f"Unknown task name: {task}")

        tk = self.tokenizer
        text_col = self.text_key[task]
        label_col = self.label_key[task]
        instr = self.task_instructions[task]

        input_max_len = self.max_input_length.get(task, max_length)
        target_max_len = self.max_target_length.get(task, max_target_length)

        src_texts = [(instr + str(t)).strip() for t in examples[text_col]]
        tgt_texts = [str(t) for t in examples[label_col]]

        if task == "CodeSearchNet":
            tgt_texts = [self._extract_first_paragraph(t) for t in tgt_texts]

        # IMPORTANT: pad to fixed lengths so PyTorch DataLoader can stack tensors
        src = tk(src_texts, padding="max_length", truncation=True, max_length=input_max_len)
        with tk.as_target_tokenizer():
            tgt = tk(tgt_texts, padding="max_length", truncation=True, max_length=target_max_len)

        labels = []
        for ids, mask in zip(tgt["input_ids"], tgt["attention_mask"]):
            labels.append([tok if m == 1 else -100 for tok, m in zip(ids, mask)])

        return {"input_ids": src["input_ids"], "attention_mask": src["attention_mask"], "labels": labels}

    def get_final_ds(self, task: str, split: str, batch_size: int, k: int = -1, seed: int = 0, return_test: bool = False, max_length: int = 512):
        if task == "CONCODE":
            dataset = load_dataset("AhmedSSoliman/CodeXGLUE-CONCODE", split=split)
        elif task == "CodeTrans":
            dataset = load_dataset("CM/codexglue_codetrans", split=split)
        elif task == "CodeSearchNet":
            dataset = load_dataset("semeru/code-text-ruby", split=split)
        elif task == "BFP":
            dataset = load_dataset("ayeshgk/code_x_glue_cc_code_refinement_annotated", split=split)

        elif task == "TheVault_Csharp":
            # NOTE: Only use the c_sharp subset and default to the smaller train split to avoid downloading everything.
            split_set = "train/small" if split == "train" else split
            dataset = load_dataset(
                "Fsoft-AIC/the-vault-function",
                languages=["c_sharp"],
                split_set=split_set,
            )

        elif task == "KodCode":
            # NOTE: Load the dataset once (train-only) and keep only Python-related samples if a language column exists.
            dataset = load_dataset("KodCode/KodCode-V1-SFT-R1", split="train")
            # Some versions expose a language/programming_language field; keep only Python to avoid unnecessary data.
            for lang_key in ("language", "lang", "programming_language"):
                if lang_key in dataset.column_names:
                    dataset = dataset.filter(lambda ex, lk=lang_key: str(ex.get(lk, "")).lower() in {"python", "py"})
                    break

        elif task == "RunBugRun":
            dataset = load_dataset("ASSERT-KTH/RunBugRun-Final", split="train")
            dataset = dataset.filter(lambda ex: ex.get("language", None) == "ruby")

        else:
            raise ValueError(f"Unknown task: {task}")

        if task in self.train_only_tasks:
            dataset = self._split_train_only(dataset, task, split, split_seed=42)

        if k != -1:
            dataset = self.select_subset_ds(dataset, k=k, seed=seed)
        else:
            dataset = dataset.shuffle(seed=seed)

        map_fn = lambda batch: self._preprocess_batch(batch, task, max_length=max_length)

        if not return_test:
            enc = dataset.map(map_fn, batched=True, remove_columns=dataset.column_names)
            enc.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            return DataLoader(enc, batch_size=batch_size, shuffle=True)
        else:
            N = len(dataset)
            ds_val = dataset.select(range(0, N // 2))
            ds_test = dataset.select(range(N // 2, N))

            outs = []
            for ds in (ds_val, ds_test):
                enc = ds.map(map_fn, batched=True, remove_columns=ds.column_names)
                enc.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
                outs.append(DataLoader(enc, batch_size=batch_size, shuffle=False))
            return outs[0], outs[1]
