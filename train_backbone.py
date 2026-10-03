#!/usr/bin/env python
"""Train the backbones of one dataset, leave one subject out.

    python train_backbone.py --dataset torus9 --backbone tinierhar --aug none --data-root /path/to/datasets
    python train_backbone.py --dataset torus9 --backbone tinierhar --aug tw40 --data-root /path/to/datasets

--aug none trains the pure backbone (no augmentation); rot / tw40 / mw40 / jo train it with rotation, time warping,
magnitude warping or jitter + offset.  Writes, per held-out subject,
    outputs/models/<dataset>/<backbone>/<aug>/subj_<N>.weights.h5 (+ .meta.json)
    outputs/backbones/<dataset>/<backbone>/<aug>/subj_<N>.json        (weighted F1 of the backbone alone)
Folds already done are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from ee_har.config import AUGMENTATIONS, BACKBONES, DATASETS, PROTOCOL

SEED = int(PROTOCOL["seed"])


def write_json(path: str, obj) -> None:
    """Write JSON atomically: a crash never leaves a half-written file that would count as a finished fold."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh, indent=1); fh.flush(); os.fsync(fh.fileno())
    os.replace(path + ".tmp", path)


def paths(out, dataset, backbone, aug, subject):
    return (os.path.join(out, "models", dataset, backbone, aug, f"subj_{subject}"),
            os.path.join(out, "backbones", dataset, backbone, aug, f"subj_{subject}.json"))


def train_fold(ds, backbone: str, aug: str, subject: int, out: str) -> dict | None:
    from sklearn.metrics import f1_score
    from tensorflow import keras
    from ee_har.augment import make_augmenter
    from ee_har.data import make_fold
    from ee_har.models import build_backbone
    from ee_har import training as T
    stem, rec_path = paths(out, ds.name, backbone, aug, subject)
    if os.path.exists(rec_path) and os.path.exists(stem + ".weights.h5"):
        return None
    t0 = time.time()
    T.set_determinism(SEED)
    fold = make_fold(ds, subject, SEED)
    model = build_backbone(fold, backbone)
    if aug == "none":
        fit = T.fit_supervised(model, fold.X_tr, fold.y_tr, fold.X_va, fold.y_va, SEED, epochs=T.EPOCHS, patience=T.PATIENCE,
                               start_from_epoch=min(T.WARMUP, max(T.EPOCHS - 1, 0)), predict_batch_size=256)
        test_batch = 256
    else:
        fit = T.fit_augmented(model, fold.X_tr, fold.y_tr, fold.X_va, fold.y_va, make_augmenter(aug, fold, SEED), SEED)
        test_batch = 64
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    model.save_weights(stem + ".weights.h5")
    meta = {"dataset": ds.name, "backbone": backbone, "aug": aug, "aug_params": AUGMENTATIONS.get(aug), "subject": int(subject),
            "seed": SEED, "epochs_run": int(fit["epochs_run"]), "val_f1": float(fit["val_f1"]),
            "scaler_mean": np.asarray(fold.mean, np.float64).tolist(), "scaler_std": np.asarray(fold.std, np.float64).tolist()}
    fd = os.open(stem + ".weights.h5", os.O_RDONLY); os.fsync(fd); os.close(fd)
    write_json(stem + ".meta.json", meta)
    pred = model.predict(fold.X_te, batch_size=test_batch, verbose=0).argmax(1)
    rec = {"dataset": ds.name, "backbone": backbone, "aug": aug, "subject": int(subject), "seed": SEED,
           "f1_weighted": float(f1_score(fold.y_te, pred, average="weighted")), "epochs_run": int(fit["epochs_run"]),
           "val_f1": float(fit["val_f1"]), "n_train": int(len(fold.y_tr)), "n_val": int(len(fold.y_va)),
           "n_test": int(len(fold.y_te)), "seconds": round(time.time() - t0, 1)}
    write_json(rec_path, rec)
    del model; keras.backend.clear_session()
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--backbone", required=True, choices=list(BACKBONES))
    ap.add_argument("--aug", required=True, choices=["none", *AUGMENTATIONS])
    ap.add_argument("--subjects", nargs="*", type=int, default=None, help="held-out subjects (default: all)")
    ap.add_argument("--data-root", default=None, help="folder with the raw datasets (or set EE_HAR_DATA)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--inter-threads", type=int, default=1)
    a = ap.parse_args()
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(a.threads); tf.config.threading.set_inter_op_parallelism_threads(a.inter_threads)
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    from ee_har.data import load_dataset
    ds = load_dataset(a.dataset, a.data_root)
    excluded = set(PROTOCOL["exclude_test_subjects"].get(a.dataset, []))
    for s in [int(s) for s in (a.subjects or np.unique(ds.subj)) if int(s) not in excluded]:
        rec = train_fold(ds, a.backbone, a.aug, s, a.out)
        if rec:
            print(f"{a.dataset}/{a.backbone}/{a.aug} subject {s} [{'GPU' if gpus else 'CPU'}]: wF1 {rec['f1_weighted']:.4f}, "
                  f"{rec['epochs_run']} epochs ({rec['seconds']:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
