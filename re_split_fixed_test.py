# re_split_fixed_test.py
import argparse, os, random, shutil, json, time
from pathlib import Path

ALLOWED = {".jpg", ".jpeg", ".png"}

def collect_images(data_root):
    by_class = {}
    for split in ["train","val","test"]:
        for class_dir in (Path(data_root)/split).glob("*"):
            if not class_dir.is_dir(): continue
            cname = class_dir.name
            for p in class_dir.iterdir():
                if p.suffix.lower() in ALLOWED:
                    by_class.setdefault(cname, []).append(p)
    return by_class

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="existing split root containing train/val/test")
    ap.add_argument("--out_root", required=True, help="new split root to create")
    ap.add_argument("--drop", nargs="*", default=[], help="classes to remove entirely")
    ap.add_argument("--test_per_class", type=int, default=20)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--move", action="store_true", help="move files instead of copying")
    args = ap.parse_args()

    random.seed(args.seed)
    by_class = collect_images(args.data_root)

    # drop classes
    for d in args.drop:
        by_class.pop(d, None)

    # sanity
    for cname, files in by_class.items():
        if len(files) < args.test_per_class:
            raise ValueError(f"class '{cname}' has only {len(files)} images (< {args.test_per_class}).")

    out = Path(args.out_root)
    for split in ["train","val","test"]:
        for cname in by_class.keys():
            (out/split/cname).mkdir(parents=True, exist_ok=True)

    manifest = {}
    for cname, files in by_class.items():
        files = list(files)
        random.shuffle(files)
        test = files[:args.test_per_class]
        rest = files[args.test_per_class:]
        vcount = max(1, int(round(args.val_ratio*len(rest))))
        val = rest[:vcount]
        train = rest[vcount:]

        manifest[cname] = {
            "train": len(train), "val": len(val), "test": len(test),
            "total": len(files)
        }

        def place(paths, split):
            for src in paths:
                dst = out/split/cname/src.name
                if args.move:
                    shutil.move(str(src), str(dst))
                else:
                    shutil.copy2(str(src), str(dst))

        place(train, "train")
        place(val,   "val")
        place(test,  "test")

    with open(out/"split_manifest.json","w") as f:
        json.dump({
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_root": args.data_root,
            "out_root": args.out_root,
            "drop": args.drop,
            "test_per_class": args.test_per_class,
            "val_ratio": args.val_ratio,
            "counts": manifest
        }, f, indent=2)

    print("Done. New split at:", out)
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
