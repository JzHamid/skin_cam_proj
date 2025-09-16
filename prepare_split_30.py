#!/usr/bin/env python3
"""
Create a 70/10/20 (train/val/test) split from data_raw/ and save 30x30 RGB images
to data_30x30/. Keeps class names from subfolder names.
"""

import argparse, random
from pathlib import Path
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(d: Path):
    return [p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS]

def letterbox_resize(img: Image.Image, size=30, fill=(0,0,0)):
    img = img.convert("RGB")
    w, h = img.size
    s = min(size/max(1,w), size/max(1,h))
    nw, nh = max(1,int(w*s)), max(1,int(h*s))
    im2 = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(im2, ((size-nw)//2, (size-nh)//2))
    return canvas

def save_split(files, out_root: Path, split: str, cname: str, size=30):
    out_c = out_root / split / cname
    out_c.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(files):
        try:
            with Image.open(src) as im:
                im2 = letterbox_resize(im, size=size)
                im2.save(out_c / f"{cname}_{i:06d}.jpg", quality=95)
        except Exception as e:
            print(f"[WARN] skip {src}: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data_raw")
    ap.add_argument("--out_dir", default="data_30x30")
    ap.add_argument("--size", type=int, default=30)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    raw, out = Path(args.raw_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    classes = sorted([d.name for d in raw.iterdir() if d.is_dir()])
    print("Classes:", classes)

    for cname in classes:
        files = list_images(raw / cname)
        files = sorted(files)
        random.shuffle(files)

        n = len(files)
        n_tr = int(n * args.train)
        n_va = int(n * args.val)
        tr = files[:n_tr]
        va = files[n_tr:n_tr+n_va]
        te = files[n_tr+n_va:]

        print(f"{cname}: total={n} | train={len(tr)} val={len(va)} test={len(te)}")

        save_split(tr, out, "train", cname, size=args.size)
        save_split(va, out, "val",   cname, size=args.size)
        save_split(te, out, "test",  cname, size=args.size)

    print("Done →", out)

if __name__ == "__main__":
    main()
