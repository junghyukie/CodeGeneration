from datasets import load_dataset
import os, json

root = os.path.dirname(__file__)
out_dir = os.path.join(root, "data", "translate")
os.makedirs(out_dir, exist_ok=True)

splits = {
    "train": "train",
    "valid": "validation",
    "test":  "test"
}
for name, hf_split in splits.items():
    ds = load_dataset("CM/codexglue_codetrans", split=hf_split)
    java, cs = ds["java"], ds["cs"]
    assert len(java) == len(cs)

    with open(os.path.join(out_dir, f"{name}.java-cs.txt.java"), "w", encoding="utf-8") as f_java, \
         open(os.path.join(out_dir, f"{name}.java-cs.txt.cs"),   "w", encoding="utf-8") as f_cs:
        for j, c in zip(java, cs):
            f_java.write(j.replace("\n", "\\n") + "\n")
            f_cs.write(c.replace("\n", "\\n") + "\n")
    print(f"✓ wrote {name} ({len(java)} pairs)")
print("ALL DONE")
