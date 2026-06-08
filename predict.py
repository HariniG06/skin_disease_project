import os
import sys
import argparse
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing import image as keras_image

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE      = 224
MODEL_PATH    = "skin_disease_model.h5"
print("current folder:", os.getcwd())
print("model exists:",os.path.exists(MODEL_PATH))
CLASSES_PATH  = "classes.json"
SUPPORTED     = {".jpg", ".jpeg", ".png", ".bmp"}

# ── Load Model & Classes ──────────────────────────────────────────────────────
_model   = None
_classes = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            sys.exit(f"[ERROR] Model not found: {MODEL_PATH}\nRun train_model.py first.")
        print(f"Loading model…")
        _model = tf.keras.models.load_model(MODEL_PATH)
        print("Model ready.\n")
    return _model

def get_classes():
    global _classes
    if _classes is None:
        if os.path.exists(CLASSES_PATH):
            with open(CLASSES_PATH) as f:
                _classes = json.load(f)
        else:
            # Auto-detect from data/test folder
            test_dir = "data/test"
            if os.path.exists(test_dir):
                _classes = sorted(os.listdir(test_dir))
            else:
                classes = [f"Class{i}" for i in range(10)]
    return _classes

# ── Preprocess ────────────────────────────────────────────────────────────────
def preprocess(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, 0)

# ── Predict One ───────────────────────────────────────────────────────────────
def predict_one(image_path: str) -> dict:
    model   = get_model()
    classes = get_classes()
    tensor  = preprocess(image_path)
    probs   = model.predict(tensor, verbose=0)[0]

    results = sorted(
        [{"name": classes[i], "confidence": float(probs[i]) * 100}
         for i in range(len(classes))],
        key=lambda x: x["confidence"],
        reverse=True,
    )
    return {"path": image_path, "top": results[0], "all": results}

# ── Print Result ──────────────────────────────────────────────────────────────
def print_result(result: dict):
    top = result["top"]
    sep = "─" * 50
    print(sep)
    print(f"  Image   : {os.path.basename(result['path'])}")
    print(f"  Result  : {top['name']}")
    print(f"  Conf.   : {top['confidence']:.1f}%")
    print()
    print("  Top-3:")
    for r in result["all"][:3]:
        bar = "█" * int(r["confidence"] / 5)
        print(f"    {r['name']:<25} {r['confidence']:5.1f}%  {bar}")
    print(sep)

# ── Visualise ─────────────────────────────────────────────────────────────────
def visualise(result: dict, save=False):
    path = result["path"]
    top  = result["top"]

    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor("#0f172a")

    ax1.imshow(img)
    ax1.axis("off")
    ax1.set_title(
        f"{top['name']}\n{top['confidence']:.1f}%",
        color="#38bdf8", fontsize=12, fontweight="bold",
    )

    ax2.set_facecolor("#1e293b")
    names = [r["name"] for r in result["all"]]
    confs = [r["confidence"] for r in result["all"]]
    colors = ["#38bdf8" if n == top["name"] else "#334155" for n in names]

    bars = ax2.barh(names, confs, color=colors)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Confidence (%)", color="#94a3b8")
    ax2.tick_params(colors="#94a3b8")
    ax2.spines[:].set_color("#334155")
    ax2.set_title("All Classes", color="#94a3b8")

    for bar, conf in zip(bars, confs):
        ax2.text(conf + 1, bar.get_y() + bar.get_height() / 2,
                 f"{conf:.1f}%", va="center", color="#e2e8f0", fontsize=8)

    plt.tight_layout()
    if save:
        out = Path(path).stem + "_result.png"
        plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
        print(f"  Saved → {out}")
    else:
        plt.show()
    plt.close()

# ── Folder Predict ────────────────────────────────────────────────────────────
def predict_folder(folder: str, save=False):
    files = [str(p) for p in Path(folder).iterdir()
             if p.suffix.lower() in SUPPORTED]
    if not files:
        print(f"No images found in: {folder}")
        return

    print(f"Found {len(files)} image(s)\n")
    for f in sorted(files):
        try:
            result = predict_one(f)
            print_result(result)
            if save:
                visualise(result, save=True)
        except Exception as e:
            print(f"  [SKIP] {f}: {e}")

# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__== "__main__":
    parser = argparse.ArgumentParser(description="Skin Disease Predictor")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str, help="Single image path")
    group.add_argument("--folder", type=str, help="Folder of images")
    parser.add_argument("--show",  action="store_true", help="Show chart")
    parser.add_argument("--save",  action="store_true", help="Save chart")
    args = parser.parse_args()

    if args.image:
        result = predict_one(args.image)
        print_result(result)
        if args.show or args.save:
            visualise(result, save=args.save)
    elif args.folder:
        predict_folder(args.folder, save=args.save)