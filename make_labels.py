from pathlib import Path

train = Path(r'.\data_raw_split_v2\train')
classes = sorted([d.name for d in train.iterdir() if d.is_dir()])

out = Path(r'.\v2_outputs\tflite\labels.txt')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(classes), encoding='utf-8')

print("Wrote", out, ":", classes)
