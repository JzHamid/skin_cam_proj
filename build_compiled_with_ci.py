#!/usr/bin/env python3
from pathlib import Path
import csv, json, re, math
from collections import defaultdict

root = Path("v2_outputs")
compiled = root / "compiled"
compiled.mkdir(parents=True, exist_ok=True)

def wilson_ci(k, n, z=1.96):
    # k successes out of n, returns (low, high)
    if n == 0:
        return (None, None)
    p = k / n
    denom = 1 + (z**2)/n
    center = (p + (z**2)/(2*n)) / denom
    halfw = z*math.sqrt((p*(1-p)/n) + (z**2)/(4*n*n)) / denom
    return (max(0.0, center - halfw), min(1.0, center + halfw))

def to_float(x):
    try:
        return float(x)
    except:
        return None

def gather_all():
    rows_overall = []     # overall rows for accuracy table
    rows_perclass = []    # per-class metrics

    for aug_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name != "compiled"]):
        aug = aug_dir.name
        for px_dir in sorted([p for p in aug_dir.iterdir() if p.is_dir() and p.name.startswith("px")]):
            m = re.search(r"px(\d+)", px_dir.name.lower())
            img_size = int(m.group(1)) if m else None

            # Try to learn n_test from any test_report.json in this px folder
            n_test = None
            sample_report = None
            for model_dir in sorted([p for p in px_dir.iterdir() if p.is_dir()]):
                rpt = model_dir / "test_report.json"
                if rpt.exists():
                    try:
                        data = json.loads(rpt.read_text(encoding="utf-8"))
                        sample_report = data
                        # sum supports over classes
                        n_test = sum(int(v.get("support", 0)) for k,v in data.items()
                                     if isinstance(v, dict) and k not in {"accuracy", "macro avg", "weighted avg"})
                        break
                    except Exception:
                        pass

            # Add overall (one row per model from results_summary.csv)
            rs = px_dir / "results_summary.csv"
            if rs.exists():
                with rs.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        p = to_float(r.get("test_acc"))
                        if p is not None and n_test:
                            k = int(round(p * n_test))
                            lo, hi = wilson_ci(k, n_test)
                        else:
                            lo = hi = None
                        rows_overall.append({
                            "InputSize": img_size,
                            "Model": r["model"],
                            "Aug": aug,
                            "TestAccuracy": p,
                            "CI95_low": lo,
                            "CI95_high": hi,
                            "ValBest": to_float(r.get("val_best")),
                            "ValEval": to_float(r.get("val_eval")),
                            "Params": r.get("params", ""),
                            "N_test": n_test
                        })

            # Per-class table (precision/recall/f1/support + CI on recall)
            for model_dir in sorted([p for p in px_dir.iterdir() if p.is_dir()]):
                rpt = model_dir / "test_report.json"
                if not rpt.exists():
                    continue
                data = json.loads(rpt.read_text(encoding="utf-8"))
                for cls, metrics in data.items():
                    if cls in {"accuracy", "macro avg", "weighted avg"}:
                        continue
                    support = int(metrics.get("support", 0) or 0)
                    recall = to_float(metrics.get("recall"))
                    k = int(round(recall * support)) if (recall is not None) else None
                    lo, hi = (None, None)
                    if k is not None and support:
                        lo, hi = wilson_ci(k, support)
                    rows_perclass.append({
                        "InputSize": img_size,
                        "Model": model_dir.name,
                        "Aug": aug,
                        "Class": cls,
                        "precision": to_float(metrics.get("precision")),
                        "recall": recall,
                        "recall_CI95_low": lo,
                        "recall_CI95_high": hi,
                        "f1": to_float(metrics.get("f1-score") or metrics.get("f1_score") or metrics.get("f1")),
                        "support": support
                    })

    return rows_overall, rows_perclass

def aggregate_repeats_with_sd(rows_overall):
    """
    If you ran multiple repeats for the same (Model, InputSize, Aug),
    compute mean and sample SD. If only one run exists, SD will be blank.
    """
    groups = defaultdict(list)
    for r in rows_overall:
        key = (r["Model"], r["InputSize"], r["Aug"])
        groups[key].append(r["TestAccuracy"])

    mean_sd = {}
    for k, vals in groups.items():
        vals = [v for v in vals if v is not None]
        if not vals:
            mean_sd[k] = (None, None)
            continue
        mean = sum(vals)/len(vals)
        if len(vals) > 1:
            var = sum((v-mean)**2 for v in vals)/(len(vals)-1)
            sd = var**0.5
        else:
            sd = None
        mean_sd[k] = (mean, sd)
    return mean_sd

def main():
    rows_overall, rows_perclass = gather_all()
    mean_sd = aggregate_repeats_with_sd(rows_overall)

    # Write overall table
    out1 = compiled / "test_acc_summary.csv"
    with out1.open("w", newline="", encoding="utf-8") as f:
        cols = ["InputSize","Model","Aug","TestAccuracy","CI95_low","CI95_high","ValBest","ValEval","Params","N_test","MeanAcc","SDAcc"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows_overall, key=lambda x:(x["Model"], x["InputSize"], x["Aug"])):
            m, sd = mean_sd[(r["Model"], r["InputSize"], r["Aug"])]
            r2 = dict(r)
            r2["MeanAcc"] = m
            r2["SDAcc"] = sd
            w.writerow(r2)

    # Write per-class table
    out2 = compiled / "per_class_summary.csv"
    with out2.open("w", newline="", encoding="utf-8") as f:
        cols = ["InputSize","Model","Aug","Class","precision","recall","recall_CI95_low","recall_CI95_high","f1","support"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows_perclass, key=lambda x:(x["Class"], x["Model"], x["InputSize"], x["Aug"])):
            w.writerow(r)

    # Also create "best per model" table for Chapter 4
    best = {}
    for r in rows_overall:
        key = r["Model"]
        cur = best.get(key)
        if (cur is None) or ((r["TestAccuracy"] or -1) > (cur["TestAccuracy"] or -1)):
            best[key] = r
    out3 = compiled / "table_best_per_model.csv"
    with out3.open("w", newline="", encoding="utf-8") as f:
        cols = ["Model","InputSize","Aug","TestAccuracy","CI95_low","CI95_high","ValBest","Params"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for model in sorted(best.keys()):
            r = best[model]
            w.writerow({c: r.get(c) for c in cols})

    print("Wrote:", out1)
    print("Wrote:", out2)
    print("Wrote:", out3)

if __name__ == "__main__":
    main()
