#!/usr/bin/env python3
"""
train_skin_compare.py

Compare 5 CNN backbones on a local skin-rash dataset stored as:
  data_30x30/
    train/<class>/*.(png|jpg|jpeg)
    val/<class>/*.(png|jpg|jpeg)
    test/<class>/*.(png|jpg|jpeg)

Disk images are 30x30; we upscale to 128x128 for all models.
Two-stage training: linear probe -> fine-tune (with min-epoch floor).

Outputs: out_skin/<model_name>/
Summary CSV: out_skin/results_summary.csv
"""

import os, csv, argparse, itertools, inspect, json
from pathlib import Path
from typing import List
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tf.keras.backend.set_image_data_format("channels_last")


# ------------------------------ Args -----------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True,
                   help="Root with train/val/test subfolders (your 30x30 dataset).")
    p.add_argument("--models",
                   default="mobilenet_v2,inception_v3,resnet50,efficientnet_b0,densenet121",
                   help="Comma-separated backbones to run.")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs_head", type=int, default=4)
    p.add_argument("--epochs_ft", type=int, default=20)
    p.add_argument("--min_epochs", type=int, default=10,
                   help="Do not early-stop before this many epochs in a stage.")
    p.add_argument("--patience", type=int, default=10,
                   help="EarlyStopping patience after min_epochs.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="out_skin")
    p.add_argument("--img_size", type=int, default=128,
               help="Target resize for loader & model input (e.g., 224 or 299).")
    return p.parse_args()


# ------------------------------ Data -----------------------------------------
def get_img_size(_backbone: str, override: int | None = None) -> int:
    # If you pass --img_size, we’ll use it for all models.
    # InceptionV3 likes >=75; 224 or 299 are common choices.
    return override if override else 128


def make_datasets(data_dir: str, img_size: int, batch: int, seed: int):
    """Load train/val/test as RGB (3 channels) and prefetch."""
    kw = dict(
        image_size=(img_size, img_size),
        batch_size=batch,
        label_mode="int",
        color_mode="rgb",   # <-- FORCE 3-CHANNEL DECODE
        seed=seed,
    )
    train = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"), shuffle=True, **kw)
    val = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "val"), shuffle=False, **kw)
    test = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "test"), shuffle=False, **kw)

    class_names = train.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    train = train.cache().prefetch(AUTOTUNE)
    val   = val.cache().prefetch(AUTOTUNE)
    test  = test.cache().prefetch(AUTOTUNE)
    return train, val, test, class_names


# ------------------------------ Model zoo ------------------------------------
def _supports_kw(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except Exception:
        return False

def build_model(backbone: str, num_classes: int, img_size: int):
    """
    Build classifier with a specific backbone.
    Use explicit input_shape=(img,img,3) so ImageNet weights match.
    """
    L = tf.keras.layers

    # Always 3 channels
    inp = L.Input((img_size, img_size, 3), name="image_rgb")

    # Light aug
    x = L.RandomFlip("horizontal", name="aug_flip")(inp)
    x = L.RandomRotation(0.05, name="aug_rot")(x)
    x = L.RandomContrast(0.10, name="aug_contrast")(x)

    bb = backbone.lower()

    if bb == "mobilenet_v2":
        x_p = L.Rescaling(1./127.5, offset=-1.0, name="scale_to_minus1_1")(x)  # [-1,1]
        base = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
        )
        feats = base(x_p, training=False)
        unfreeze_n = 30

    elif bb == "inception_v3":
        x_p = L.Lambda(tf.keras.applications.inception_v3.preprocess_input,
                       name="pre_inception")(x)
        base = tf.keras.applications.InceptionV3(
            include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
        )
        feats = base(x_p, training=False)
        unfreeze_n = 50

    elif bb == "resnet50":
        x_p = L.Lambda(tf.keras.applications.resnet50.preprocess_input,
                       name="pre_resnet")(x)
        base = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
        )
        feats = base(x_p, training=False)
        unfreeze_n = 50

    elif bb == "efficientnet_b0":
        # Preprocess to match EfficientNet’s expectations
        x_p = L.Lambda(tf.keras.applications.efficientnet.preprocess_input,
                    name="pre_effnet")(x)

        Eff = tf.keras.applications.EfficientNetB0
        pretrain_source = "imagenet"

        # 1) Try the standard way first (no input_tensor/input_shape here)
        try:
            base = Eff(include_top=False, weights="imagenet")
        except Exception as e:
            print("[WARN] EfficientNet direct ImageNet load failed:", e)
            pretrain_source = "partial_imagenet_skip_mismatch"

            # 2) Build the model with no weights, then manually load with skip_mismatch=True
            base = Eff(include_top=False, weights=None)
            try:
                wpath = tf.keras.utils.get_file(
                    "efficientnetb0_notop.h5",
                    origin="https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5",
                    cache_subdir="models",
                )
                # Loads all matching weights; mismatched are skipped
                base.load_weights(wpath, by_name=False, skip_mismatch=True)
                print("[INFO] EfficientNet: partial ImageNet weights loaded (skip_mismatch).")
            except Exception as e2:
                print("[WARN] EfficientNet manual partial load failed; using random init.", e2)
                pretrain_source = "random"

        # 3) Now apply the base to OUR tensor
        feats = base(x_p, training=False)
        unfreeze_n = 60  # how many top layers to unfreeze in fine-tuning

        # optional debug
        try:
            print(f"[DEBUG] efficientnet_b0: base.input_shape={base.input_shape} | pretrain={pretrain_source}")
        except Exception:
            pass

    elif bb == "densenet121":
        x_p = L.Lambda(tf.keras.applications.densenet.preprocess_input,
                       name="pre_densenet")(x)
        base = tf.keras.applications.DenseNet121(
            include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
        )
        feats = base(x_p, training=False)
        unfreeze_n = 50

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    x = L.GlobalAveragePooling2D(name="gap")(feats)
    x = L.Dropout(0.30, name="head_dropout")(x)
    out = L.Dense(num_classes, activation="softmax", name="classifier")(x)
    model = tf.keras.Model(inp, out, name=f"{bb}_classifier")

    try:
        print(f"[DEBUG] {bb}: base.input_shape={base.input_shape}")
    except Exception:
        pass

    return model, base, unfreeze_n


# ------------------------------ Training utils --------------------------------
def eval_split(model, dataset, class_names: List[str]):
    y_true, y_pred = [], []
    for xb, yb in dataset:
        pr = model.predict(xb, verbose=0)
        y_pred.extend(np.argmax(pr, axis=1))
        y_true.extend(yb.numpy())
    rep_txt = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    acc = (np.trace(cm) / np.sum(cm)) if np.sum(cm) else 0.0
    return rep_txt, cm, float(acc), y_true, y_pred

def plot_confusion(cm: np.ndarray, class_names: List[str], out_path: Path, title="Confusion Matrix"):
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title); plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    thresh = cm.max() / 2. if cm.size else 0.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], "d"),
                 ha="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True"); plt.xlabel("Predicted"); plt.tight_layout()
    plt.savefig(out_path, dpi=160); plt.close()

def plot_history(hist, out_dir: Path, title_prefix=""):
    plt.figure()
    plt.plot(hist["accuracy"], label="train_acc")
    plt.plot(hist["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"{title_prefix} Accuracy per Epoch"); plt.legend()
    plt.tight_layout(); plt.savefig(out_dir/"history_accuracy.png", dpi=160); plt.close()

    plt.figure()
    plt.plot(hist["loss"], label="train_loss")
    plt.plot(hist["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{title_prefix} Loss per Epoch"); plt.legend()
    plt.tight_layout(); plt.savefig(out_dir/"history_loss.png", dpi=160); plt.close()

class EarlyStopWithFloor(tf.keras.callbacks.Callback):
    """Don’t early-stop before min_epochs. After that, apply patience on val_accuracy."""
    def __init__(self, monitor="val_accuracy", mode="max", patience=10, min_epochs=10, restore_best_weights=True):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_epochs = min_epochs
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.best = -np.inf if mode == "max" else np.inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        cur = logs.get(self.monitor)
        if cur is None:
            return
        improve = (cur > self.best) if self.mode == "max" else (cur < self.best)
        if improve:
            self.best = cur
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            if epoch + 1 >= self.min_epochs:
                self.wait += 1
                if self.wait >= self.patience:
                    if self.restore_best_weights and self.best_weights is not None:
                        self.model.set_weights(self.best_weights)
                    self.model.stop_training = True


# ------------------------------ Main ------------------------------------------
def main():
    args = parse_args()
    tf.random.set_seed(args.seed); np.random.seed(args.seed)
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)

    wanted = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    summary_rows = [["model","img_size","params","val_best","val_eval","test_acc"]]

    for backbone in wanted:
        img_size = get_img_size(backbone, args.img_size)
        train_ds, val_ds, test_ds, class_names = make_datasets(args.data_dir, img_size, args.batch, args.seed)
        num_classes = len(class_names)

        model, base, unfreeze_n = build_model(backbone, num_classes, img_size)
        out_dir = out_root / backbone
        out_dir.mkdir(parents=True, exist_ok=True)

        # ----- Stage A: Linear probe -----
        base.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=["accuracy"])
        cb_head = [
            EarlyStopWithFloor(monitor="val_accuracy", mode="max",
                               patience=args.patience, min_epochs=args.min_epochs, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(str(out_dir/"ckpt_head.keras"),
                                               monitor="val_accuracy", save_best_only=True)
        ]
        hist_head = model.fit(train_ds, validation_data=val_ds,
                              epochs=max(args.epochs_head, args.min_epochs),
                              callbacks=cb_head, verbose=2)

        try:
            model = tf.keras.models.load_model(str(out_dir/"ckpt_head.keras"))
        except Exception:
            pass

        # ----- Stage B: Fine-tune -----
        for layer in base.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
        if unfreeze_n and unfreeze_n < len(base.layers):
            for l in base.layers[:-unfreeze_n]:
                l.trainable = False

        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=["accuracy"])
        cb_ft = [
            EarlyStopWithFloor(monitor="val_accuracy", mode="max",
                               patience=args.patience, min_epochs=args.min_epochs, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(str(out_dir/"ckpt_ft.keras"),
                                               monitor="val_accuracy", save_best_only=True)
        ]
        hist_ft = model.fit(train_ds, validation_data=val_ds,
                            epochs=max(args.epochs_ft, args.min_epochs),
                            callbacks=cb_ft, verbose=2)

        try:
            model = tf.keras.models.load_model(str(out_dir/"ckpt_ft.keras"))
        except Exception:
            pass

        # ----- Evaluation & reports -----
        # Validation
        rep_val, cm_val, val_acc, y_true_val, y_pred_val = eval_split(model, val_ds, class_names)
        (out_dir/"validation_report.txt").write_text(
            rep_val + "\nConfusion matrix:\n" + np.array2string(cm_val), encoding="utf-8"
        )
        # NEW: JSON dumps (val)
        val_report = classification_report(
            y_true_val, y_pred_val, target_names=class_names, output_dict=True, zero_division=0
        )
        (out_dir/"validation_report.json").write_text(json.dumps(val_report, indent=2), encoding="utf-8")
        with open(out_dir/"val_confusion_matrix.json","w", encoding="utf-8") as f:
            json.dump({"labels": class_names, "cm": cm_val.tolist()}, f, indent=2)

        # Test
        rep_test, cm_test, test_acc, y_true_test, y_pred_test = eval_split(model, test_ds, class_names)
        (out_dir/"test_report.txt").write_text(
            rep_test + "\nConfusion matrix:\n" + np.array2string(cm_test), encoding="utf-8"
        )
        # NEW: JSON dumps (test)
        test_report = classification_report(
            y_true_test, y_pred_test, target_names=class_names, output_dict=True, zero_division=0
        )
        (out_dir/"test_report.json").write_text(json.dumps(test_report, indent=2), encoding="utf-8")
        with open(out_dir/"test_confusion_matrix.json","w", encoding="utf-8") as f:
            json.dump({"labels": class_names, "cm": cm_test.tolist()}, f, indent=2)

        # ----- Plots -----
        hist_all = {
            "accuracy":     hist_head.history.get("accuracy", [])     + hist_ft.history.get("accuracy", []),
            "val_accuracy": hist_head.history.get("val_accuracy", []) + hist_ft.history.get("val_accuracy", []),
            "loss":         hist_head.history.get("loss", [])         + hist_ft.history.get("loss", []),
            "val_loss":     hist_head.history.get("val_loss", [])     + hist_ft.history.get("val_loss", []),
        }
        plot_history(hist_all, out_dir, title_prefix=backbone)
        plot_confusion(cm_val,  class_names, out_dir/"confusion_val.png",  title=f"{backbone} Validation CM")
        plot_confusion(cm_test, class_names, out_dir/"confusion_test.png", title=f"{backbone} Test CM")

        val_best = max(hist_all["val_accuracy"]) if hist_all["val_accuracy"] else float("nan")
        summary_rows.append([backbone, str(img_size), f"{model.count_params():,}",
                             f"{val_best:.4f}", f"{val_acc:.4f}", f"{test_acc:.4f}"])

        print(f"\n[RESULT] {backbone}: val_best={val_best:.4f} | val_eval={val_acc:.4f} | test={test_acc:.4f}")
        print(f"Saved outputs under: {out_dir}")

    with open(out_root/"results_summary.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(summary_rows)
    print("\nWrote summary CSV:", out_root/"results_summary.csv")


if __name__ == "__main__":
    main()
