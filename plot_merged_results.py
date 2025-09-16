#!/usr/bin/env python3
"""
plot_merged_results.py

Reads: out_skin/ALL_results_summary.csv
Writes:
  out_skin/plots/all_models_test_acc_vs_size.png
  out_skin/plots/test_acc_heatmap.png
  out_skin/plots/best_by_model.csv
"""

from pathlib import Path
import csv
import math

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

SRC = Path("out_skin") / "ALL_results_summary.csv"
PLOTS_DIR = Path("out_skin") / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Load & clean ----------
rows = []
if not SRC.exists():
    raise SystemExit(f"CSV not found: {SRC}. Run merge_results.py first.")

with SRC.open("r", newline="", encoding="utf-8") as fh:
    rdr = csv.DictReader(fh)
    for r in rdr:
        try:
            r["model"] = r["model"].strip().lower()
            # coerce numerics
            r["img_size"]  = int(float(r["img_size"]))
            r["val_best"]  = float(r["val_best"])
            r["val_eval"]  = float(r["val_eval"])
            r["test_acc"]  = float(r["test_acc"])
            rows.append(r)
        except Exception:
            # skip malformed lines
            continue

if not rows:
    raise SystemExit("No valid rows parsed from CSV.")

# ---------- Aggregate: best test_acc per (model, img_size) ----------
best_by_key = {}  # (model, img_size) -> row
for r in rows:
    key = (r["model"], r["img_size"])
    cur = best_by_key.get(key)
    if cur is None or r["test_acc"] > cur["test_acc"]:
        best_by_key[key] = r

# Organize by model
by_model = {}
for (m, s), r in best_by_key.items():
    by_model.setdefault(m, []).append(r)

# ---------- Plot: Test accuracy vs image size (one line per model) ----------
plt.figure(figsize=(9, 6))
for m, lst in sorted(by_model.items()):
    lst_sorted = sorted(lst, key=lambda x: x["img_size"])
    xs = [r["img_size"] for r in lst_sorted]
    ys = [r["test_acc"] for r in lst_sorted]
    plt.plot(xs, ys, marker="o", label=m)

plt.xlabel("Input image size (pixels)")
plt.ylabel("Test accuracy")
plt.title("Test accuracy vs. input size (per model)")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()
out_path1 = PLOTS_DIR / "all_models_test_acc_vs_size.png"
plt.tight_layout()
plt.savefig(out_path1, dpi=160)
plt.close()

# ---------- Heatmap: models x sizes of test_acc ----------
# Collect sorted unique sizes and models
all_sizes = sorted({r["img_size"] for r in best_by_key.values()})
all_models = sorted(by_model.keys())

# Build matrix with NaN for missing combos
def get(m, s):
    rr = best_by_key.get((m, s))
    return rr["test_acc"] if rr else float("nan")

mat = [[get(m, s) for s in all_sizes] for m in all_models]

plt.figure(figsize=(1 + 0.5*len(all_sizes), 1 + 0.5*len(all_models)))
im = plt.imshow(mat, aspect="auto", interpolation="nearest")
plt.colorbar(im)
plt.xticks(range(len(all_sizes)), all_sizes, rotation=45, ha="right")
plt.yticks(range(len(all_models)), all_models)
plt.title("Test accuracy heatmap (model × size)")
plt.xlabel("Input size (px)")
plt.ylabel("Model")

# Annotate each cell with value
for i in range(len(all_models)):
    for j in range(len(all_sizes)):
        v = mat[i][j]
        if not math.isnan(v):
            plt.text(j, i, f"{v:.2f}", ha="center", va="center")

out_path2 = PLOTS_DIR / "test_acc_heatmap.png"
plt.tight_layout()
plt.savefig(out_path2, dpi=160)
plt.close()

# ---------- Best size per model CSV ----------
best_by_model = []
for m, lst in by_model.items():
    top = max(lst, key=lambda r: r["test_acc"])
    gap = top["test_acc"] - top["val_eval"]  # positive means test > val_eval
    best_by_model.append({
        "model": m,
        "best_size": top["img_size"],
        "test_acc": f"{top['test_acc']:.4f}",
        "val_eval": f"{top['val_eval']:.4f}",
        "test_minus_valeval": f"{gap:.4f}",
        "params": top.get("params", "")
    })

best_csv = PLOTS_DIR / "best_by_model.csv"
with best_csv.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(best_by_model[0].keys()))
    w.writeheader()
    w.writerows(best_by_model)

print("Saved:")
print(" -", out_path1)
print(" -", out_path2)
print(" -", best_csv)
