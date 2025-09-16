#!/usr/bin/env python3
import csv
from pathlib import Path

OUT_ROOT = Path("out_skin")
out_csv  = OUT_ROOT / "ALL_results_summary.csv"

rows = []
header = None

# pick up any results_summary.csv under out_skin/** (e.g., px128, px160, ...)
for f in OUT_ROOT.rglob("results_summary.csv"):
    run_dir = f.parent.name  # e.g. px128
    with f.open("r", newline="", encoding="utf-8") as fh:
        r = list(csv.reader(fh))
        if not r:
            continue
        h = r[0]
        if header is None:
            # add a column that shows which run folder (px size batch) the row came from
            header = h + ["run_folder"]
        for row in r[1:]:
            rows.append(row + [run_dir])

# Write merged CSV
OUT_ROOT.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    if header is None:
        header = ["model","img_size","params","val_best","val_eval","test_acc","run_folder"]
    w.writerow(header)
    w.writerows(rows)

print("Merged", len(rows), "rows into", out_csv)
