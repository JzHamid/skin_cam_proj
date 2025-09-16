#!/usr/bin/env python3
from pathlib import Path
from PIL import Image
import argparse, shutil

OK_FORMATS = {"JPEG", "PNG", "GIF", "BMP"}  # what tf.io.decode_image supports

def sniff_format(p: Path):
    try:
        with Image.open(p) as im:
            im.verify()      # integrity
        with Image.open(p) as im2:
            return im2.format  # e.g. "JPEG", "PNG", "WEBP", ...
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--quarantine", default="_quarantine")
    args = ap.parse_args()

    root = Path(args.root)
    qdir = root / args.quarantine
    qdir.mkdir(exist_ok=True)

    moved = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        fmt = sniff_format(p)
        if fmt is None or fmt.upper() not in OK_FORMATS:
            # corrupt or unsupported (e.g., WEBP/HEIC masquerading as .png/.jpg)
            dst = qdir / p.name
            print(f"[MOVE] {p}  ->  {dst}   (fmt={fmt})")
            shutil.move(str(p), str(dst))
            moved += 1

    print(f"\nQuarantined {moved} files to {qdir}")
if __name__ == "__main__":
    main()
