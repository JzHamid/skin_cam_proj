# make_per_class_summary.py
import json, argparse
from pathlib import Path
import pandas as pd

def collect_reports(root: Path):
    rows = []
    for aug_dir in root.iterdir():
        if not aug_dir.is_dir(): 
            continue
        aug = aug_dir.name
        for size_dir in aug_dir.iterdir():
            if not size_dir.is_dir(): 
                continue
            px = size_dir.name.replace("px","")
            for model_dir in size_dir.iterdir():
                if not model_dir.is_dir(): 
                    continue
                model = model_dir.name
                rep = model_dir / "test_report.json"
                if not rep.exists(): 
                    continue
                try:
                    data = json.loads(rep.read_text())
                except Exception:
                    continue
                # Expect sklearn-like classification_report dict
                for cls, stats in data.items():
                    if cls in ("accuracy", "macro avg", "weighted avg"): 
                        continue
                    if not isinstance(stats, dict): 
                        continue
                    rows.append({
                        "augmentation": aug,
                        "px": int(px),
                        "model": model,
                        "class": cls,
                        "precision": stats.get("precision"),
                        "recall": stats.get("recall"),
                        "f1": stats.get("f1-score") or stats.get("f1"),
                        "support": stats.get("support"),
                    })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = collect_reports(Path(args.root))
    df = df.sort_values(["augmentation","model","px","class"])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out} with {len(df)} rows")

if __name__ == "__main__":
    main()
