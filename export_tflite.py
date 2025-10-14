#!/usr/bin/env python3
import argparse, pathlib, time, os, random
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

# ---------- helpers ----------
def representative_dataset_gen(img_dir, img_size, max_samples=100):
    if img_dir is None:
        return
    exts = {".jpg",".jpeg",".png"}
    files = []
    for root, _, fns in os.walk(img_dir):
        for f in fns:
            if os.path.splitext(f.lower())[1] in exts:
                files.append(os.path.join(root, f))
    random.shuffle(files)
    for p in files[:max_samples]:
        try:
            im = Image.open(p).convert("RGB").resize((img_size, img_size))
            x = np.asarray(im, dtype=np.float32)[None, ...]  # (1,H,W,3) 0..255
            yield [x]
        except Exception:
            continue

def _load_with_preprocess_fix(ckpt_path: str):
    """Robust loader for .keras that used Lambda(preprocess_input)."""
    ckpt_lower = ckpt_path.lower()

    # Candidate preprocess functions by backbone
    pre_effnet    = tf.keras.applications.efficientnet.preprocess_input
    pre_inception = tf.keras.applications.inception_v3.preprocess_input
    pre_resnet    = tf.keras.applications.resnet50.preprocess_input
    pre_densenet  = tf.keras.applications.densenet.preprocess_input

    # Heuristic: pick based on path; fall back to trying them all, then identity.
    ordered = []
    if "inception"   in ckpt_lower: ordered = [pre_inception]
    elif "resnet"    in ckpt_lower: ordered = [pre_resnet]
    elif "densenet"  in ckpt_lower: ordered = [pre_densenet]
    elif "efficient" in ckpt_lower or "effnet" in ckpt_lower: ordered = [pre_effnet]
    else:
        ordered = [pre_inception, pre_resnet, pre_densenet, pre_effnet]

    # 1) Try raw load first (may work e.g., MobileNet which used Rescaling layer)
    try:
        return tf.keras.models.load_model(ckpt_path, compile=False)
    except Exception as e:
        last = e

    # 2) Try with the guessed preprocess
    for fn in ordered:
        try:
            return tf.keras.models.load_model(
                ckpt_path, compile=False, custom_objects={"preprocess_input": fn}
            )
        except Exception as e2:
            last = e2

    # 3) Final fallback: identity (only if all else fails)
    try:
        return tf.keras.models.load_model(
            ckpt_path, compile=False, custom_objects={"preprocess_input": lambda x: x}
        )
    except Exception:
        raise last

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to trained Keras model (.keras)")
    ap.add_argument("--img_size", type=int, required=True, help="Input size H=W")
    ap.add_argument("--out", required=True, help="Output .tflite path")
    ap.add_argument("--quantize", choices=["none","dynamic","float16","int8"], default="dynamic")
    ap.add_argument("--calib_dir", default=None, help="(INT8) directory of images for calibration (e.g., train/)")
    ap.add_argument("--calib_samples", type=int, default=100, help="(INT8) number of calibration images")
    ap.add_argument("--select_tf_ops", action="store_true",
                    help="Allow SELECT_TF_OPS fallback if converter requires it")
    args = ap.parse_args()

    model = _load_with_preprocess_fix(args.ckpt)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if args.select_tf_ops:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
        ]

    if args.quantize == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif args.quantize == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif args.quantize == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if args.calib_dir is None:
            raise SystemExit("INT8 requires --calib_dir pointing to sample images.")
        converter.representative_dataset = lambda: representative_dataset_gen(
            args.calib_dir, args.img_size, args.calib_samples
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.uint8
        converter.inference_output_type = tf.uint8

    tfl = converter.convert()
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_bytes(tfl)
    print(f"Saved: {outp}  ({outp.stat().st_size/1024:.1f} KB)")

    # Quick latency smoke test on random input
    inter = tf.lite.Interpreter(model_path=str(outp))
    inter.allocate_tensors()
    inp = inter.get_input_details()[0]
    out = inter.get_output_details()[0]
    x = (np.random.rand(1, args.img_size, args.img_size, 3).astype(np.float32))*255.0

    for _ in range(3):  # warmup
        inter.set_tensor(inp["index"], x); inter.invoke()

    iters = 30
    t0 = time.perf_counter()
    for _ in range(iters):
        inter.set_tensor(inp["index"], x); inter.invoke()
    t1 = time.perf_counter()
    print(f"Avg TFLite latency: {((t1 - t0)/iters)*1000:.2f} ms / image")

if __name__ == "__main__":
    main()
