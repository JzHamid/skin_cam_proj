import csv, re
from pathlib import Path

root = Path("out_skin")
rows = [["run_folder","img_size","model","params","val_best","val_eval","test_acc"]]
for sub in sorted(root.glob("px*/results_summary.csv")):
    m = re.search(r"px(\d+)", str(sub))
    img_size = m.group(1) if m else ""
    with sub.open("r", encoding="utf-8") as f:
        r = list(csv.reader(f))
    # skip header row in each summary
    for row in r[1:]:
        rows.append([sub.parent.name, img_size] + row)

with (root/"master_results.csv").open("w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(rows)

print("Wrote", root/"master_results.csv")
