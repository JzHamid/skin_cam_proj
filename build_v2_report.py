#!/usr/bin/env python3
# build_v2_report.py — merge all runs, make plots, rank mobile picks
import os, re, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional: only import TF when timing inference to keep memory lower
try:
    import tensorflow as tf
except Exception:
    tf = None

def _load_ds(split_dir, img_size, batch=32, seed=42):
    return tf.keras.utils.image_dataset_from_directory(
        split_dir, image_size=(img_size, img_size), batch_size=batch,
        label_mode="int", color_mode="rgb", shuffle=False, seed=seed)

def measure_inference_ms_per_image(ckpt_path, img_size, data_dir, batch=32):
    if tf is None:
        return None
    test_dir = Path(data_dir) / "test"
    test = _load_ds(str(test_dir), img_size, batch=batch)
    n_images = 0
    for _, y in test: n_images += y.shape[0]
    test = _load_ds(str(test_dir), img_size, batch=batch)  # reload after count
    model = tf.keras.models.load_model(str(ckpt_path))
    # warmup
    for xb, _ in test.take(1):
        _ = model.predict(xb, verbose=0)
    t0 = time.perf_counter()
    for xb, _ in test:
        _ = model.predict(xb, verbose=0)
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000.0
    return total_ms / max(1, n_images)

def scan_runs(root, data_dir):
    rows = []
    for cond in ["noaug", "aug_light", "aug_medium"]:
        cond_dir = Path(root) / cond
        if not cond_dir.exists(): continue
        for px_dir in sorted(cond_dir.glob("px*")):
            m = re.search(r"px(\d+)", px_dir.name)
            if not m: continue
            img_size = int(m.group(1))
            for model_dir in sorted(px_dir.iterdir()):
                if not model_dir.is_dir(): continue
                test_json = model_dir / "test_report.json"
                if not test_json.exists(): continue
                try:
                    with open(test_json, "r", encoding="utf-8") as f:
                        tj = json.load(f)
                except Exception:
                    continue
                macro = tj.get("macro avg", {})
                weighted = tj.get("weighted avg", {})
                accuracy = float(tj.get("accuracy", 0.0))
                support_total = sum(v["support"] for k, v in tj.items()
                                    if isinstance(v, dict) and "support" in v and k not in ["macro avg","weighted avg"])
                ckpt_ft = model_dir / "ckpt_ft.keras"
                params, inf_ms = None, None
                # params + latency if we have TF and a checkpoint
                if tf is not None and ckpt_ft.exists():
                    try:
                        mdl = tf.keras.models.load_model(str(ckpt_ft))
                        params = int(mdl.count_params())
                        del mdl
                    except Exception:
                        pass
                    try:
                        inf_ms = measure_inference_ms_per_image(ckpt_ft, img_size, data_dir)
                    except Exception:
                        pass
                rows.append({
                    "condition": cond,
                    "img_size": img_size,
                    "model": model_dir.name,
                    "accuracy": accuracy,
                    "precision_macro": macro.get("precision", np.nan),
                    "recall_macro": macro.get("recall", np.nan),
                    "f1_macro": macro.get("f1-score", np.nan),
                    "precision_weighted": weighted.get("precision", np.nan),
                    "recall_weighted": weighted.get("recall", np.nan),
                    "f1_weighted": weighted.get("f1-score", np.nan),
                    "n_test": support_total,
                    "inference_ms_per_image": inf_ms,
                    "params": params,
                    "run_dir": str(model_dir),
                    "ckpt_ft": str(ckpt_ft) if ckpt_ft.exists() else "",
                })
    return pd.DataFrame(rows)

def heatmap(df, cond, out_png):
    sub = df[df["condition"]==cond]
    if sub.empty: return
    models = sorted(sub["model"].unique())
    sizes  = sorted(sub["img_size"].unique())
    grid = np.full((len(models), len(sizes)), np.nan)
    for i, m in enumerate(models):
        for j, s in enumerate(sizes):
            q = sub[(sub.model==m) & (sub.img_size==s)]
            if not q.empty:
                grid[i, j] = q["accuracy"].max()
    plt.figure(figsize=(12,5))
    im = plt.imshow(grid, aspect="auto")
    plt.colorbar(im, fraction=0.025)
    plt.xticks(range(len(sizes)), sizes, rotation=40)
    plt.yticks(range(len(models)), models)
    for i in range(len(models)):
        for j in range(len(sizes)):
            if not np.isnan(grid[i,j]):
                plt.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title(f"Test accuracy heatmap — {cond}")
    plt.xlabel("Input size (px)"); plt.ylabel("Model")
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

def lines_by_model(df, cond, out_png):
    sub = df[df["condition"]==cond]
    if sub.empty: return
    plt.figure(figsize=(10,6))
    for m, g in sub.groupby("model"):
        g = g.sort_values("img_size")
        plt.plot(g["img_size"], g["accuracy"], marker="o", label=m)
    plt.legend(ncol=2); plt.xlabel("Input size (px)"); plt.ylabel("Test accuracy")
    plt.title(f"Accuracy vs. input size — {cond}")
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

def bar_best_per_model(df, cond, out_png):
    sub = df[df["condition"]==cond]
    if sub.empty: return
    best = sub.sort_values(["model","accuracy","img_size"], ascending=[True,False,True]) \
              .groupby("model", as_index=False).first()
    best = best.sort_values("accuracy", ascending=False)
    plt.figure(figsize=(9,6))
    plt.bar(best["model"], best["accuracy"])
    for i, r in best.iterrows():
        plt.text(i, r["accuracy"]+0.005, f"px{int(r['img_size'])}", ha="center", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Test accuracy"); plt.title(f"Best accuracy per model — {cond}")
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

def rank_mobile(df):
    d = df.dropna(subset=["accuracy","inference_ms_per_image","params"]).copy()
    if d.empty: return d
    d["acc_z"] = (d["accuracy"] - d["accuracy"].mean()) / d["accuracy"].std(ddof=0)
    d["lat_z"] = (d["inference_ms_per_image"] - d["inference_ms_per_image"].mean()) / d["inference_ms_per_image"].std(ddof=0)
    d["par_z"] = (d["params"] - d["params"].mean()) / d["params"].std(ddof=0)
    d["mobile_score"] = (2.0 * d["acc_z"]) - (0.8 * d["lat_z"]) - (0.6 * d["par_z"])
    return d.sort_values("mobile_score", ascending=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="v2_outputs", help="Folder with noaug/aug_light/… subdirs")
    ap.add_argument("--data_dir", required=True, help="Root with train/val/test")
    args = ap.parse_args()

    root = Path(args.root); (root/"plots").mkdir(parents=True, exist_ok=True)
    df = scan_runs(root, args.data_dir)
    out_csv = root/"all_results_v2.csv"
    df.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)

    if df.empty:
        print("No rows found. Check that your runs wrote test_report.json files.")
        return

    for cond in df["condition"].dropna().unique():
        heatmap(df, cond, root/"plots"/f"heatmap_{cond}.png")
        lines_by_model(df, cond, root/"plots"/f"acc_vs_size_{cond}.png")
        bar_best_per_model(df, cond, root/"plots"/f"best_per_model_{cond}.png")

    ranked = rank_mobile(df)
    if not ranked.empty:
        ranked.to_csv(root/"best_for_mobile.csv", index=False)
        top = ranked.head(5)
        txt = ["Top mobile-ready configs (score ↓):"]
        for _, r in top.iterrows():
            txt.append(f"- {r['condition']} | {r['model']} @ px{int(r['img_size'])} | "
                       f"acc={r['accuracy']:.3f}, ms/img={r['inference_ms_per_image']:.1f}, "
                       f"params={int(r['params']):,}")
        (root/"plots"/"top_mobile_picks.txt").write_text("\n".join(txt), encoding="utf-8")
        print("\n".join(txt))

if __name__ == "__main__":
    main()
