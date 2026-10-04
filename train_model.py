"""Train the CNN and save model, preprocessor and evaluation metrics into ./models.

Usage:
    python train_model.py                          # uses data/sample_train.csv
    python train_model.py --data KDDTrain+.txt     # real NSL-KDD file
"""
import argparse
import json
import os

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from preprocess import CLASSES, Preprocessor, load_csv, map_label

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(BASE, "data", "sample_train.csv"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        print("Training file not found; generating sample data first...")
        import generate_sample_data
        generate_sample_data.main()

    import tensorflow as tf
    from tensorflow import keras
    from model import build_model

    tf.keras.utils.set_random_seed(42)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Loading {args.data}")
    df = load_csv(args.data)
    if "label" not in df.columns:
        raise SystemExit("The training file needs a label column.")
    df = df.drop_duplicates().reset_index(drop=True)
    df["target"] = df["label"].map(map_label)
    dropped = int(df["target"].isna().sum())
    if dropped:
        print(f"Dropping {dropped} rows with unrecognised labels")
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    y = df["target"].map(CLASSES.index).astype(int).values
    print("Class counts:", {c: int((y == i).sum()) for i, c in enumerate(CLASSES)})

    idx = np.arange(len(df))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
    va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=42)

    pre = Preprocessor().fit(df.iloc[tr])
    X_tr, X_va, X_te = (pre.transform(df.iloc[i]) for i in (tr, va, te))
    y_tr, y_va, y_te = y[tr], y[va], y[te]

    present = np.unique(y_tr)
    weights = compute_class_weight("balanced", classes=present, y=y_tr)
    class_weight = {int(c): float(w) for c, w in zip(present, weights)}

    model = build_model(X_tr.shape[1], len(CLASSES))
    model.summary()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=args.epochs,
                     batch_size=args.batch, class_weight=class_weight, callbacks=callbacks, verbose=2)

    pred = model.predict(X_te, batch_size=1024, verbose=0).argmax(axis=1)
    labels = list(range(len(CLASSES)))
    p, r, f, _ = precision_recall_fscore_support(y_te, pred, average="weighted", zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(p), "recall": float(r), "f1": float(f),
        "classes": CLASSES,
        "confusion_matrix": confusion_matrix(y_te, pred, labels=labels).tolist(),
        "per_class": classification_report(y_te, pred, labels=labels, target_names=CLASSES,
                                           output_dict=True, zero_division=0),
        "history": {k: [float(v) for v in vals] for k, vals in hist.history.items()
                    if k in ("loss", "val_loss", "accuracy", "val_accuracy")},
        "samples": {"train": len(tr), "validation": len(va), "test": len(te)},
        "trained_on": os.path.basename(args.data),
    }
    model.save(os.path.join(MODEL_DIR, "nids_cnn.keras"))
    joblib.dump(pre, os.path.join(MODEL_DIR, "preprocessor.joblib"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nTest accuracy {metrics['accuracy']:.4f} | precision {p:.4f} | recall {r:.4f} | F1 {f:.4f}")
    print("Saved model, preprocessor and metrics to", MODEL_DIR)


if __name__ == "__main__":
    main()
