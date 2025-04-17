from datasets import load_dataset
import json, os, sys

root_dir = os.path.dirname(__file__)
out_dir  = os.path.join(root_dir, "data", "concode")
os.makedirs(out_dir, exist_ok=True)

splits = {
    "train": "train",
    "dev":   "validation",
    "test":  "test"
}

for out_name, hf_split in splits.items():
    print(f"Downloading Concode {hf_split} …")
    ds = load_dataset("AhmedSSoliman/CodeXGLUE-CONCODE", split=hf_split)
    fn = os.path.join(out_dir, f"{out_name}.json")
    with open(fn, "w", encoding="utf-8") as f:
        for nl, code in zip(ds["nl"], ds["code"]):
            f.write(json.dumps({"nl": nl, "code": code}, ensure_ascii=False) + "\n")
    print(f"  → saved {fn}  lines={len(ds)}")

print("✓ Done!")