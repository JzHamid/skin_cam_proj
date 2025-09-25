# build_report.py
import os, json, argparse, datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def find_runs(out_root):
    out = []
    for pxdir in sorted(Path(out_root).glob("px*")):
        try:
            size = int(pxdir.name.replace("px",""))
        except:
            continue
        for model_dir in pxdir.iterdir():
            if not model_dir.is_dir():
                continue
            rpt = model_dir/"test_report.json"
            cmj = model_dir/"test_confusion_matrix.json"
            if rpt.exists():
                out.append({"size": size, "model": model_dir.name,
                            "report": rpt, "cmjson": cmj if cmj.exists() else None})
    return out

def load_metrics(run):
    data = json.loads(run["report"].read_text())
    overall = {
        "img_size": run["size"],
        "model": run["model"],
        "accuracy": float(data.get("accuracy", np.nan)),
        "macro_f1": float(data.get("macro avg", {}).get("f1-score", np.nan)),
        "weighted_f1": float(data.get("weighted avg", {}).get("f1-score", np.nan)),
    }
    per_class = []
    for cname, d in data.items():
        if cname in ["accuracy","macro avg","weighted avg"]:
            continue
        per_class.append({
            "img_size": run["size"], "model": run["model"], "class": cname,
            "precision": float(d.get("precision", np.nan)),
            "recall": float(d.get("recall", np.nan)),
            "f1": float(d.get("f1-score", np.nan)),
            "support": int(d.get("support", 0)),
        })
    return overall, per_class

def plot_heatmap(df_overall, out_dir):
    pivot = df_overall.pivot(index="model", columns="img_size", values="accuracy").sort_index()
    plt.figure()
    plt.imshow(pivot.values, aspect="auto")
    plt.xticks(range(pivot.shape[1]), pivot.columns, rotation=45)
    plt.yticks(range(pivot.shape[0]), pivot.index)
    plt.colorbar(label="Test Accuracy")
    plt.title("Accuracy heatmap (model × img_size)")
    plt.tight_layout()
    plt.savefig(out_dir/"heatmap_accuracy.png", dpi=200)
    plt.close()

def plot_lines(df_overall, out_dir):
    plt.figure()
    for model, g in df_overall.groupby("model"):
        g = g.sort_values("img_size")
        plt.plot(g["img_size"], g["accuracy"], marker="o", label=model)
    plt.xlabel("Input size (px)")
    plt.ylabel("Test accuracy")
    plt.title("Accuracy vs input size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir/"accuracy_vs_size.png", dpi=200)
    plt.close()

def best_size_by_model(df_overall, metric="accuracy"):
    # pick smallest size within 0.01 of absolute best (knee-friendly)
    best = {}
    for model, g in df_overall.groupby("model"):
        g = g.sort_values("img_size")
        m = g[metric].max()
        cand = g[g[metric] >= (m - 0.01)]
        best[model] = int(cand.iloc[0]["img_size"])  # smallest size within 1% of best
    return best

def perclass_bars(df_perclass, out_dir, best_sizes):
    for model, size in best_sizes.items():
        sub = df_perclass[(df_perclass["model"]==model) & (df_perclass["img_size"]==size)]
        sub = sub.sort_values("f1", ascending=False)
        plt.figure()
        plt.bar(sub["class"], sub["f1"])
        plt.ylim(0,1)
        plt.xticks(rotation=40, ha="right")
        plt.title(f"Per-class F1 — {model} @ {size}px")
        plt.tight_layout()
        plt.savefig(out_dir/f"{model}_best_{size}_perclass_f1.png", dpi=200)
        plt.close()

def winners_per_class(df_perclass, df_overall, out_dir):
    # for each class, select (model,size) giving best F1
    df = df_perclass.copy()
    idx = df.sort_values(["class","f1","img_size"], ascending=[True,False,True]) \
            .groupby("class").head(1)
    idx = idx[["class","model","img_size","f1","precision","recall","support"]]
    idx.to_csv(out_dir/"winners_per_class.csv", index=False)
    return idx

def class_difficulty(df_perclass, best_sizes, out_dir):
    # take each model at its chosen best size; average per-class F1 across models
    frames = []
    for model, size in best_sizes.items():
        frames.append(df_perclass[(df_perclass["model"]==model) & (df_perclass["img_size"]==size)])
    cat = pd.concat(frames, ignore_index=True)
    agg = cat.groupby("class", as_index=False)["f1"].mean().rename(columns={"f1":"mean_f1_at_best"})
    agg.sort_values("mean_f1_at_best", ascending=True).to_csv(out_dir/"class_difficulty.csv", index=False)

def confusion_insights(runs, out_dir, best_sizes):
    rows = []
    for r in runs:
        model = r["model"]; size = r["size"]
        if model not in best_sizes or best_sizes[model] != size or r["cmjson"] is None:
            continue
        j = json.loads(r["cmjson"].read_text())
        labels = j["labels"]; cm = np.array(j["cm"], dtype=np.int32)
        # top 5 off-diagonal confusions
        off = []
        for i in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                if i==k: continue
                if cm[i,k]>0: off.append((labels[i], labels[k], int(cm[i,k])))
        off.sort(key=lambda t: t[2], reverse=True)
        for src, dst, cnt in off[:5]:
            rows.append({"model": model, "img_size": size, "from": src, "to": dst, "count": cnt})
    if rows:
        pd.DataFrame(rows).to_csv(out_dir/"top_confusions.csv", index=False)

def write_summary_md(df_overall, best_sizes, winners, out_dir):
    lines = []
    lines.append("# Skin Cam — Results Report\n")
    lines.append("## Highlights\n")
    lines.append("- **Best size per model** (smallest size within 1% of that model’s best accuracy):")
    for m,s in best_sizes.items():
        acc = df_overall[(df_overall.model==m) & (df_overall.img_size==s)]["accuracy"].iloc[0]
        lines.append(f"  - **{m}** @ **{s}px** — acc: **{acc:.3f}**")
    lines.append("\n- **Per-class winners** (best F1): see `winners_per_class.csv`.\n")
    lines.append("## Plots\n")
    lines.append("- `accuracy_vs_size.png`")
    lines.append("- `heatmap_accuracy.png`")
    lines.append("- `*_perclass_f1.png` (each model at its chosen best size)\n")
    (out_dir/"README.md").write_text("\n".join(lines), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out_skin", help="root folder containing px*/model runs")
    ap.add_argument("--out_dir", default="", help="report output folder (default: out_skin/report_YYYYMMDD_HHMM)")
    args = ap.parse_args()

    runs = find_runs(args.root)
    if not runs:
        print("No runs found.")
        return

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.root)/f"report_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_rows, perclass_rows = [], []
    for r in runs:
        o, pc = load_metrics(r)
        overall_rows.append(o)
        perclass_rows.extend(pc)

    df_overall = pd.DataFrame(overall_rows)
    df_perclass = pd.DataFrame(perclass_rows)

    df_overall.to_csv(out_dir/"overall_metrics.csv", index=False)
    df_perclass.to_csv(out_dir/"perclass_metrics.csv", index=False)

    plot_lines(df_overall, out_dir)
    plot_heatmap(df_overall, out_dir)
    best = best_size_by_model(df_overall, metric="accuracy")
    perclass_bars(df_perclass, out_dir, best)
    winners = winners_per_class(df_perclass, df_overall, out_dir)
    class_difficulty(df_perclass, best, out_dir)
    confusion_insights(runs, out_dir, best)
    write_summary_md(df_overall, best, winners, out_dir)

    print(f"Report written to: {out_dir}")

if __name__ == "__main__":
    main()
