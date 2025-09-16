#!/usr/bin/env python3
import argparse, shutil, random
from pathlib import Path

IMG_EXT = {".jpg",".jpeg",".png",".bmp",".gif",".tif",".tiff",".webp"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="data_raw with class folders only")
    ap.add_argument("--dst", required=True, help="output root with train/val/test")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val",   type=float, default=0.10)
    ap.add_argument("--test",  type=float, default=0.20)
    ap.add_argument("--seed",  type=int, default=42)
    ap.add_argument("--move", action="store_true", help="move instead of copy")
    args = ap.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, "splits must sum to 1"
    random.seed(args.seed)

    src = Path(args.src); dst = Path(args.dst)
    for s in ("train","val","test"): (dst/s).mkdir(parents=True, exist_ok=True)

    classes = [d for d in src.iterdir() if d.is_dir()]
    if not classes: raise SystemExit(f"No class folders in {src}")

    for c in sorted(classes):
        imgs = [p for p in c.rglob("*") if p.suffix.lower() in IMG_EXT]
        if not imgs:
            print(f"[WARN] no images in {c.name}"); continue
        random.shuffle(imgs)
        n = len(imgs); n_tr = int(round(args.train*n)); n_va = int(round(args.val*n))
        buckets = [("train", imgs[:n_tr]),
                   ("val",   imgs[n_tr:n_tr+n_va]),
                   ("test",  imgs[n_tr+n_va:])]
        for split, lst in buckets:
            out = dst/split/c.name; out.mkdir(parents=True, exist_ok=True)
            for p in lst:
                (shutil.move if args.move else shutil.copy2)(str(p), str(out/p.name))
            print(f"[{c.name}] {split}: {len(lst)} -> {out}")
    print("Done:", dst)

if __name__ == "__main__":
    main()
