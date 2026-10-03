#!/usr/bin/env python
"""Fit the early exits of one augmentation and evaluate the 2 x 2 study, leave one subject out.

    python train_exits.py --dataset torus9 --backbone tinierhar --aug tw40 --data-root /path/to/datasets

Needs the pure backbone (train_backbone.py --aug none) and the backbone trained with the augmentation (--aug tw40).
For each of the two backbones, fits exit heads on clean windows and on augmented windows, chooses the thresholds on
clean and on augmented validation windows for every lambda, and scores the cascade on the held-out subject.  Writes
outputs/exits/<dataset>/<backbone>/<aug>/subj_<N>.json:
    backbones.{pure, aug}.cells.{clean, aug}.thresholds.{clean, aug}[i] -> wF1, cost (mac_full, flop_full), exit shares
    (backbone x exit heads x thresholds; the paper's 2 x 2 uses thresholds "clean")
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import numpy as np

from ee_har.config import AUGMENTATIONS, BACKBONES, DATASETS, PROTOCOL, STUDY

SEED = int(STUDY["seed"])


def run_fold(ds, backbone: str, aug: str, subject: int, out: str) -> dict | None:
    from tensorflow import keras
    from ee_har.data import make_fold
    from ee_har.early_exit import cell, extract
    from ee_har.models import build_backbone
    from ee_har.training import set_determinism
    rec_path = os.path.join(out, "exits", ds.name, backbone, aug, f"subj_{subject}.json")
    if os.path.exists(rec_path):
        return None
    t0 = time.time()
    rec = {"dataset": ds.name, "backbone": backbone, "aug": aug, "aug_params": AUGMENTATIONS[aug], "subject": int(subject),
           "seed": SEED, "test_seed": int(STUDY["test_seed"]), "copies": int(STUDY["copies"]), "backbones": {}}
    for kind, trained_with in (("pure", "none"), ("aug", aug)):
        stem = os.path.join(out, "models", ds.name, backbone, trained_with, f"subj_{subject}")
        if not os.path.exists(stem + ".weights.h5"):
            raise SystemExit(f"missing {stem}.weights.h5: run train_backbone.py --aug {trained_with} first")
        set_determinism(SEED)
        fold = make_fold(ds, subject, SEED)
        model = build_backbone(fold, backbone)
        model.load_weights(stem + ".weights.h5")
        E = extract(fold, model, backbone, aug)
        rec["backbones"][kind] = {"ecost": E["ecost"], "full": E["full"], "cells": {hk: cell(E, hk) for hk in ("clean", "aug")}}
        rec.update(K=E["K"], n_te=E["n_te"], tap=E["tap"])
        del E, model; keras.backend.clear_session(); gc.collect()
    rec["seconds"] = round(time.time() - t0, 1)
    from train_backbone import write_json
    write_json(rec_path, rec)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--backbone", required=True, choices=list(BACKBONES))
    ap.add_argument("--aug", required=True, choices=list(AUGMENTATIONS))
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
    lam = float(STUDY["deployed_lambda"])
    for s in [int(s) for s in (a.subjects or np.unique(ds.subj)) if int(s) not in excluded]:
        rec = run_fold(ds, a.backbone, a.aug, s, a.out)
        if rec:
            row = lambda k, h: next(r for r in rec["backbones"][k]["cells"][h]["thresholds"]["clean"] if r["lambda"] == lam)
            print(f"{a.dataset}/{a.backbone}/{a.aug} subject {s} [{'GPU' if gpus else 'CPU'}], lambda {lam}: " +
                  "  ".join(f"{k} backbone + {h} heads {row(k, h)['wf1']:.3f} @ {100 * row(k, h)['flop_full']:.0f} % FLOPs"
                            for k in ("pure", "aug") for h in ("clean", "aug")) + f"  ({rec['seconds']:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
