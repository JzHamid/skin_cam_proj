#!/usr/bin/env python3
"""
build_v2_stats.py

Aggregates ALL runs under v2_outputs/ and computes:
- all_results_v2.csv  (one row per aug × size × model)
- per_class_results_v2.csv  (one row per class per run)
- stats_by_model_aug.csv    (for each aug+model: mean±std across sizes)
- stats_by_size_aug.csv     (for each aug+size: mean±std across models)
- aug_gain.csv              (acc augmentation deltas per model×size)
- per_class_stats_by_model_aug.csv (for each aug+model+class: mean±std F1 across sizes)

Also generates a few PNG plots into v2_outputs/plots_v2/

Requires: pandas, matplotlib
"""

import argparse, json, re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="v2_outputs",
                   help="Root folder containing augmentation dirs: noaug/ aug_light/ ...")
    p.add_argument("--out", default=None,
                   help="Where to write CSVs/plots. Default is <root>.")
    return p.parse_args()


PX_RE = re.compile(r"^px(\d+)$")

def _px_of(p: Path) -> Optional[int]:
    m = PX_RE.match(p.name.lower())
    return int(m.group(1)) if m else None


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def load_one_run(aug_dir: Path, px_dir: Path, model_dir: Path) -> Optional[Dict]:
    """
    Read one run folder:
      <root>/<aug>/px{N}/{model}/
    Expect test_report.json; optionally parse params/val metrics from px-level results_summary.csv
    """
    out = {
        "aug": aug_dir.name,
        "img_size": _px_of(px_dir),
        "model": model_dir.name,
        "params": np.nan,
        "val_best": np.nan,
        "val_eval": np.nan,
        "test_acc": np.nan,
        "macro_precision": np.nan,
        "macro_recall": np.nan,
        "macro_f1": np.nan,
    }

    # --- test_report.json (authoritative per-class + accuracy) ---
    tr_json = model_dir / "test_report.json"
    if not tr_json.exists():
        # Fallback: attempt to infer accuracy from the text file if JSON missing
        tr_txt = model_dir / "test_report.txt"
        if tr_txt.exists():
            try:
                txt = tr_txt.read_text(encoding="utf-8", errors="ignore")
                # crude parse: last "accuracy" line in sklearn report
                for line in txt.splitlines():
                    if line.strip().startswith("accuracy"):
                        parts = line.split()
                        # e.g., ['accuracy', '', '0.8603', '229']
                        for token in parts:
                            f = _safe_float(token)
                            if 0.0 <= f <= 1.0:
                                out["test_acc"] = f
                                break
                # per-class not available from txt reliably; skip
            except Exception:
                pass
        return out  # either with or without test_acc

    try:
        data = json.loads(tr_json.read_text(encoding="utf-8"))
    except Exception:
        return out

    # accuracy is a scalar at root
    if isinstance(data.get("accuracy"), (int, float)):
        out["test_acc"] = float(data["accuracy"])

    # macro avg dict
    macro = data.get("macro avg", {})
    out["macro_precision"] = _safe_float(macro.get("precision"))
    out["macro_recall"]    = _safe_float(macro.get("recall"))
    out["macro_f1"]        = _safe_float(macro.get("f1-score"))

    # --- params/val metrics from px-level summary (optional) ---
    px_summary = px_dir / "results_summary.csv"
    if px_summary.exists():
        try:
            dfpx = pd.read_csv(px_summary)
            # expected header: model,img_size,params,val_best,val_eval,test_acc
            # find row where model matches
            row = dfpx.loc[dfpx["model"].str.lower() == model_dir.name.lower()]
            if not row.empty:
                r = row.iloc[0]
                out["params"]   = r.get("params", np.nan)
                out["val_best"] = _safe_float(r.get("val_best", np.nan))
                out["val_eval"] = _safe_float(r.get("val_eval", np.nan))
                # if test_acc missing from JSON, backfill from px summary
                if np.isnan(out["test_acc"]):
                    out["test_acc"] = _safe_float(r.get("test_acc", np.nan))
        except Exception:
            pass

    return out


def load_one_run_per_class(aug_dir: Path, px_dir: Path, model_dir: Path) -> List[Dict]:
    """
    Build per-class rows from test_report.json:
      aug, img_size, model, class, precision, recall, f1, support
    """
    rows = []
    tr_json = model_dir / "test_report.json"
    if not tr_json.exists():
        return rows
    try:
        data = json.loads(tr_json.read_text(encoding="utf-8"))
    except Exception:
        return rows

    for k, v in data.items():
        # skip summary keys
        if k in ("accuracy", "macro avg", "weighted avg"):
            continue
        if not isinstance(v, dict):
            continue
        rows.append({
            "aug": aug_dir.name,
            "img_size": _px_of(px_dir),
            "model": model_dir.name,
            "class": k,
            "precision": _safe_float(v.get("precision")),
            "recall":    _safe_float(v.get("recall")),
            "f1":        _safe_float(v.get("f1-score")),
            "support":   int(v.get("support", 0)) if str(v.get("support","")).isdigit() else np.nan,
        })
    return rows


def gather_all(root: Path):
    all_rows = []
    per_class_rows = []

    for aug_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        # expect aug names like "noaug", "aug_light", etc.
        for px_dir in sorted([d for d in aug_dir.iterdir() if d.is_dir() and _px_of(d) is not None],
                             key=lambda p: _px_of(p)):
            for model_dir in sorted([d for d in px_dir.iterdir() if d.is_dir()]):
                run = load_one_run(aug_dir, px_dir, model_dir)
                if run is not None:
                    all_rows.append(run)
                per_class_rows.extend(load_one_run_per_class(aug_dir, px_dir, model_dir))

    df_all = pd.DataFrame(all_rows).dropna(subset=["img_size"])
    df_all["img_size"] = df_all["img_size"].astype(int)
    df_all.sort_values(["aug","model","img_size"], inplace=True)

    df_cls = pd.DataFrame(per_class_rows).dropna(subset=["img_size"])
    if not df_cls.empty:
        df_cls["img_size"] = df_cls["img_size"].astype(int)
        df_cls.sort_values(["aug","model","class","img_size"], inplace=True)

    return df_all, df_cls


def compute_stats(df_all: pd.DataFrame, df_cls: pd.DataFrame, out_dir: Path):
    # ----- 1) Save raw tables -----
    out_all = out_dir / "all_results_v2.csv"
    df_all.to_csv(out_all, index=False)

    out_cls = out_dir / "per_class_results_v2.csv"
    df_cls.to_csv(out_cls, index=False)

    # ----- 2) Means & STDs -----
    # (A) For each aug+model: over sizes
    gb = df_all.groupby(["aug","model"], as_index=False)
    stats_model = gb.agg(
        mean_test_acc = ("test_acc","mean"),
        std_test_acc  = ("test_acc","std"),
        n_sizes       = ("img_size","nunique"),
        mean_macro_f1 = ("macro_f1","mean"),
        std_macro_f1  = ("macro_f1","std"),
    )

    # best size per model+aug
    idx = df_all.groupby(["aug","model"])["test_acc"].idxmax()
    best = df_all.loc[idx, ["aug","model","img_size","test_acc"]].rename(
        columns={"img_size":"best_size","test_acc":"best_acc"}
    )
    stats_model = stats_model.merge(best, on=["aug","model"], how="left")
    stats_model.to_csv(out_dir/"stats_by_model_aug.csv", index=False)

    # (B) For each aug+img_size: over models
    stats_size = df_all.groupby(["aug","img_size"], as_index=False).agg(
        mean_test_acc = ("test_acc","mean"),
        std_test_acc  = ("test_acc","std"),
        n_models      = ("model","nunique")
    )
    stats_size.to_csv(out_dir/"stats_by_size_aug.csv", index=False)

    # (C) Augmentation deltas per model×size (aug_light vs noaug if both exist)
    if set(df_all["aug"].unique()) >= {"noaug","aug_light"}:
        piv = df_all.pivot_table(index=["model","img_size"], columns="aug", values="test_acc", aggfunc="first")
        if "aug_light" in piv.columns and "noaug" in piv.columns:
            piv["delta_aug_light_minus_noaug"] = piv["aug_light"] - piv["noaug"]
        piv.reset_index().to_csv(out_dir/"aug_gain.csv", index=False)

    # (D) Per-class mean±std F1 across sizes for each aug+model+class
    if not df_cls.empty:
        per_class_stats = df_cls.groupby(["aug","model","class"], as_index=False).agg(
            mean_f1=("f1","mean"),
            std_f1=("f1","std"),
            n_sizes=("img_size","nunique")
        )
        per_class_stats.to_csv(out_dir/"per_class_stats_by_model_aug.csv", index=False)

    # ----- 3) Plots -----
    plots = out_dir/"plots_v2"
    plots.mkdir(parents=True, exist_ok=True)

    # Heatmap of accuracy for each aug
    for aug, df_sub in df_all.groupby("aug"):
        if df_sub.empty: 
            continue
        # pivot: models x sizes -> accuracy
        sizes = sorted(df_sub["img_size"].unique())
        models = sorted(df_sub["model"].unique())
        mat = np.full((len(models), len(sizes)), np.nan)
        for i, m in enumerate(models):
            row = df_sub[df_sub["model"]==m].set_index("img_size")["test_acc"]
            for j, s in enumerate(sizes):
                mat[i,j] = row.get(s, np.nan)

        plt.figure(figsize=(12, 5 + 0.3*len(models)))
        im = plt.imshow(mat, aspect="auto")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(sizes)), sizes, rotation=45)
        plt.yticks(range(len(models)), models)
        plt.title(f"Test accuracy heatmap — {aug}")
        plt.xlabel("Input size (px)")
        plt.ylabel("Model")
        # annotate sparsely
        for i in range(len(models)):
            for j in range(len(sizes)):
                val = mat[i,j]
                if np.isfinite(val):
                    plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(plots/f"heatmap_test_acc_{aug}.png", dpi=160)
        plt.close()

    # Bar: mean±std acc per model for each aug
    for aug, dfm in stats_model.groupby("aug"):
        if dfm.empty: 
            continue
        order = dfm.sort_values("mean_test_acc", ascending=False)
        x = np.arange(len(order))
        plt.figure(figsize=(10, 5))
        plt.bar(x, order["mean_test_acc"])
        plt.errorbar(x, order["mean_test_acc"], yerr=order["std_test_acc"], fmt="none", ecolor="black", capsize=3)
        plt.xticks(x, order["model"], rotation=30, ha="right")
        plt.ylim(0, 1.0)
        plt.ylabel("Mean test accuracy")
        plt.title(f"Mean ± SD test accuracy across sizes — {aug}")
        plt.tight_layout()
        plt.savefig(plots/f"bar_mean_sd_acc_{aug}.png", dpi=160)
        plt.close()

    # Per-class F1 (mean across sizes) heatmap for each aug
    if not df_cls.empty:
        pcs = df_cls.groupby(["aug","model","class"], as_index=False).agg(mean_f1=("f1","mean"))
        for aug, sub in pcs.groupby("aug"):
            models = sorted(sub["model"].unique())
            classes = sorted(sub["class"].unique())
            mat = np.full((len(models), len(classes)), np.nan)
            for i, m in enumerate(models):
                row = sub[sub["model"]==m].set_index("class")["mean_f1"]
                for j, c in enumerate(classes):
                    mat[i,j] = row.get(c, np.nan)
            plt.figure(figsize=(12, 5 + 0.3*len(models)))
            im = plt.imshow(mat, aspect="auto", vmin=0, vmax=1)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
            plt.yticks(range(len(models)), models)
            plt.title(f"Per-class mean F1 (averaged across sizes) — {aug}")
            plt.xlabel("Class")
            plt.ylabel("Model")
            for i in range(len(models)):
                for j in range(len(classes)):
                    val = mat[i,j]
                    if np.isfinite(val):
                        plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
            plt.tight_layout()
            plt.savefig(plots/f"per_class_f1_mean_{aug}.png", dpi=160)
            plt.close()


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = Path(args.out or args.root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Scanning runs under: {root}")
    df_all, df_cls = gather_all(root)
    if df_all.empty:
        print("[WARN] No runs found.")
        return

    print(f"[INFO] Found {len(df_all)} runs; building tables and plots...")
    compute_stats(df_all, df_cls, out_dir)
    print(f"[DONE] Wrote CSVs into: {out_dir}")
    print(f"[DONE] Plots saved under: {out_dir / 'plots_v2'}")


if __name__ == "__main__":
    main()
