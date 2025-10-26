# gather_test_acc.py
import argparse, csv
from pathlib import Path
import pandas as pd

def find_result_csvs(root: Path):
    # looks for ...\aug_*\px*\results_summary.csv
    for aug_dir in root.iterdir():
        if not aug_dir.is_dir(): 
            continue
        for px_dir in aug_dir.iterdir():
            if not px_dir.is_dir(): 
                continue
            f = px_dir / "results_summary.csv"
            if f.exists():
                yield aug_dir.name, px_dir.name, f

def load_one(csv_path: Path):
    # expected header: model,img_size,params,val_best,val_eval,test_acc
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        # Some versions might not have header—try manual parse
        rows = list(csv.reader(csv_path.read_text().splitlines()))
        df = pd.DataFrame(rows[1:], columns=rows[0])
    # coerce numerics
    for c in ["img_size","val_best","val_eval","test_acc"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out",  required=True)
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for aug, px, p in find_result_csvs(root):
        df = load_one(p)
        for _, r in df.iterrows():
            if str(r.get("model")) == "model":  # skip header row if read weird
                continue
            rows.append({
                "augmentation": aug,
                "px": int(str(px).replace("px","")),
                "model": str(r.get("model")),
                "img_size": r.get("img_size"),
                "val_best": r.get("val_best"),
                "val_eval": r.get("val_eval"),
                "test_acc": r.get("test_acc"),
            })
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows).dropna(subset=["test_acc"])
    raw.to_csv(out_path, index=False)

    # mean/std per (augmentation, model)
    stats = (raw
             .groupby(["augmentation","model"])["test_acc"]
             .agg(["mean","std","count"])
             .reset_index()
             .sort_values(["augmentation","mean"], ascending=[True, False]))
    stats_dir = out_path.parent / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(stats_dir / "test_acc_model_stats.csv", index=False)

    print(f"Wrote raw:  {out_path}  ({len(raw)} rows)")
    print(f"Wrote stats: {stats_dir/'test_acc_model_stats.csv'}")

if __name__ == "__main__":
    main()
