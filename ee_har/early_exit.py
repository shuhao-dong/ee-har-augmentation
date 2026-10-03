"""The early-exit pipeline on a frozen backbone, and the 2 x 2 study of one augmentation.

* Exits: three taps per backbone (models.EXIT_TAPS); each tap output is pooled over time as [mean; max; std] and fed to
  one softmax layer (an exit head); the fourth exit is the backbone's own classifier.
* Exit heads are fitted on the frozen backbone (Adam 1e-3, up to 200 epochs, early stopping on the validation loss
  with patience 10), either on the clean training windows or on 4 augmented copies of every training window (early
  stopping on 4 augmented copies of the validation windows).
* Stopping rule: a window leaves at the first early exit whose softmax entropy is below that exit's threshold.  The
  thresholds are chosen on validation windows from a grid of exit shares (0-0.8 per exit, step 0.1) to maximise
  agreement with the final exit minus lambda x the expected full cost (MAC accounting, as in the paper's runs; every
  record also reports the same routing priced in FLOPs).
* Cost: backbone FLOPs from the TF profiler (+ the recurrent layers analytically), plus the pooling and exit heads of
  every early exit evaluated on the way.
"""
from __future__ import annotations

import gc
import itertools
from typing import Sequence

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from .augment import make_augmenter
from .config import EXIT_HEAD, PROTOCOL, STUDY
from .models import EXIT_TAPS

BATCH: int = int(PROTOCOL["batch_size"])
SEED, TEST_SEED, COPIES = int(STUDY["seed"]), int(STUDY["test_seed"]), int(STUDY["copies"])
LAMBDAS, SHARE_GRID = list(STUDY["lambdas"]), list(STUDY["share_grid"])
POOL_OPS_PER_ELEMENT = int(STUDY["pool_ops_per_element"])
POOL_WIDTH = 3                                            # [mean; max; std]


# ------------------------------------------------------------------------------------------------ taps and costs
def pool_tap(F: np.ndarray) -> np.ndarray:
    """Pool a tap output over time ([N, T, C] or [N, T, C, F] -> [N, 3C]) as [mean; max; std]; a 2-D array is
    already pooled and is returned as float32."""
    F = np.asarray(F, np.float32)
    if F.ndim == 2:
        return F
    Z = F.reshape(F.shape[0], F.shape[1], -1)
    return np.concatenate([Z.mean(axis=1), Z.max(axis=1), Z.std(axis=1)], axis=1)


def tap_model(model: keras.Model, student: str) -> keras.Model:
    """Sub-model of the backbone returning [tap1, tap2, tap3, penultimate, probs] for the taps of EXIT_TAPS."""
    outs = [model.get_layer(t).output for t in EXIT_TAPS[student]]
    return keras.Model(model.input, outs + [model.get_layer("softmax").input, model.output])


def model_flops(model: keras.Model, input_shape: Sequence[int]) -> float:
    """TF profiler float_ops of a Keras model for one window (recurrent layers are added by gru_flops)."""
    f = tf.function(lambda x: model(x)).get_concrete_function(tf.TensorSpec([1] + list(input_shape), tf.float32))
    g = convert_variables_to_constants_v2(f).graph
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    opts["output"] = "none"
    return float(tf.compat.v1.profiler.profile(graph=g, run_meta=tf.compat.v1.RunMetadata(), cmd="op", options=opts).total_float_ops)


def gru_flops(model: keras.Model) -> float:
    """Analytic FLOPs of every (bidirectional) GRU or LSTM layer, which the profiler misses:
    ndir * T * 2 * gates * (d_in * u + u * u), gates = 3 (GRU) or 4 (LSTM)."""
    tot = 0.0
    for l in model.layers:
        inner = getattr(l, "forward_layer", None) or (l if isinstance(l, (keras.layers.GRU, keras.layers.LSTM)) else None)
        if inner is None:
            continue
        T, d_in = int(l.input.shape[1]), int(l.input.shape[-1])
        u = int(inner.units)
        gates = 4 if isinstance(inner, keras.layers.LSTM) else 3
        ndir = 2 if getattr(l, "backward_layer", None) is not None else 1
        tot += ndir * T * 2.0 * gates * (d_in * u + u * u)
    return tot


def exit_costs(model: keras.Model, backbone: str, input_shape: Sequence[int]) -> dict:
    """FLOPs per window for the full backbone and up to each tap, including that exit's head (2 * D * K)."""
    full = model_flops(model, input_shape) + gru_flops(model)
    n_classes = int(model.output.shape[-1])
    costs = []
    for t in EXIT_TAPS[backbone]:
        sub = keras.Model(model.input, model.get_layer(t).output)
        d = int(np.prod(sub.output.shape[2:])) * POOL_WIDTH                     # the head's input width after pooling
        costs.append(model_flops(sub, input_shape) + gru_flops(sub) + 2.0 * d * n_classes)
    return {"full_flops": full, "exit_flops": costs}


def cost_per_exit_flops(ecost, full, tap, K, mask) -> np.ndarray:
    """Full cost in FLOPs of leaving at exit 0..3 under ``mask`` (early exits evaluated), as a share of one full
    inference: backbone up to the exit + pooling (4 FLOPs per element) and head of every early exit evaluated."""
    D = [3 * t["per_step"] for t in tap]
    pool = [POOL_OPS_PER_ELEMENT * t["T"] * t["per_step"] for t in tap]
    f, head = float(full), [2.0 * D[k] * K for k in range(3)]
    trunk = [ecost[k] - head[k] for k in range(3)]
    out = []
    for k in range(4):
        ev = [j for j in range(min(k + 1, 3)) if mask[j]]
        out.append(((trunk[k] if k < 3 else f) + sum(pool[j] + head[j] for j in ev)) / f)
    return np.asarray(out)


def pooled_taps(tapm, X, n_taps, chunk=512):
    """Pooled taps [(N, D_k)] and the penultimate (N, d) of X, pooled chunk by chunk (the raw taps of a whole split
    can exceed the memory of a 16 GB machine on DSADS)."""
    Z = [[] for _ in range(n_taps)]; P = []
    for i in range(0, len(X), chunk):
        out = tapm.predict(X[i:i + chunk], batch_size=64, verbose=0)
        for k in range(n_taps):
            Z[k].append(pool_tap(out[k]))
        P.append(np.asarray(out[n_taps], np.float32)); del out
    return [np.concatenate(z) for z in Z], np.concatenate(P)


def views(tapm, n_taps, augment, Xs_list, copies=COPIES, chunk=256):
    """``copies`` augmented views of each split in ``Xs_list``, in order, from ONE stateful augmenter (the training
    draws first, then validation), pooled chunk by chunk.  Copies are stacked copy by copy.  Returns
    [(taps, penultimate)] per split."""
    out = []
    for Xs in Xs_list:
        Z = [[] for _ in range(n_taps)]; P = []
        for _ in range(copies):
            for i in range(0, len(Xs), chunk):
                o = tapm.predict(augment(tf.constant(Xs[i:i + chunk])).numpy(), batch_size=BATCH, verbose=0)
                for k in range(n_taps):
                    Z[k].append(pool_tap(o[k]))
                P.append(np.asarray(o[n_taps], np.float32)); del o
        out.append(([np.concatenate(z, 0) for z in Z], np.concatenate(P, 0)))
    return out


# ------------------------------------------------------------------------------------------------ exit heads
def fit_exit_head(Ztr, y_tr, Zva, y_va, n_classes: int, seed: int = SEED):
    """One exit head (a softmax layer) on pooled tap features; returns (kernel, bias) as float64."""
    Ztr, Zva = pool_tap(Ztr), pool_tap(Zva)
    keras.utils.set_random_seed(seed)
    ic = keras.Sequential([keras.layers.Input((Ztr.shape[1],)),
                           keras.layers.Dense(n_classes, activation="softmax", name="softmax")])
    ic.compile(optimizer=keras.optimizers.Adam(EXIT_HEAD["ic_lr"]), loss="sparse_categorical_crossentropy")
    ic.fit(Ztr, y_tr, validation_data=(Zva, y_va), epochs=EXIT_HEAD["ic_epochs"], batch_size=BATCH, verbose=0, shuffle=True,
           callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=EXIT_HEAD["ic_patience"], restore_best_weights=True)])
    d = ic.get_layer("softmax")
    return d.kernel.numpy().astype(np.float64), d.bias.numpy().astype(np.float64)


def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def entropy(p):
    return -(p * np.log(p + 1e-12)).sum(-1)


def wf1(y, pred):
    from sklearn.metrics import f1_score
    return float(f1_score(y, pred, average="weighted"))


class Heads:
    """The four exits as (W, b): exit heads 1-3 and the backbone's own classifier."""
    def __init__(self, Wb):
        self.W = [np.array(W, np.float64) for W, _ in Wb]; self.b = [np.array(b, np.float64) for _, b in Wb]

    def prob(self, k, F):
        return softmax(F @ self.W[k] + self.b[k])


# ------------------------------------------------------------------------------------------------ thresholds
def cost_vectors(ecost, full, D, K, pool, combining):
    """Per mask: normalised FULL cost of leaving at exit 0..3 (share of one full-model inference without exits)."""
    full_m = full / 2.0
    head = [D[k] * K for k in range(3)]
    trunk = [ecost[k] / 2.0 - head[k] for k in range(3)]
    vec = {}
    for mask in itertools.product([False, True], repeat=3):
        c = []
        for k in range(4):
            ev = [j for j in range(min(k + 1, 3)) if mask[j]]  # early exits evaluated before / at k
            extra = sum(pool[j] + head[j] for j in ev)
            n_eval = len(ev) + (1 if k == 3 else 0)
            comb = 2 * K * max(n_eval - 1, 0) if combining else 0
            c.append(((trunk[k] if k < 3 else full_m) + extra + comb) / full_m)
        vec[mask] = np.asarray(c, np.float64)
    return vec


def search(Hs, As, masks_of, Eold_vec, Efull_by_mask, grid):
    """All share triples of the grid in itertools.product order, sequential entropy quantiles on the
    calibration windows.  Returns per triple: taus, agreement with the final output, E_old, E_full, mask."""
    out = []
    n = len(Hs[(True, True, True)][0])
    for shares in itertools.product(grid, repeat=3):
        mask = tuple(s > 0 for s in shares)
        H, A = Hs[masks_of(mask)], As[masks_of(mask)]
        remaining = np.arange(n); u = np.full(n, 3); t = []
        for k in range(3):
            if len(remaining) == 0 or shares[k] == 0:
                t.append(None); continue
            tau = float(np.quantile(H[k][remaining], shares[k])); t.append(tau)
            take = H[k][remaining] < tau; u[remaining[take]] = k; remaining = remaining[~take]
        dec = A[u, np.arange(n)]
        agree = float(np.mean(dec == A[3]))
        E_old = float(np.mean(np.asarray([Eold_vec[k] for k in u])))
        E_full = float(np.mean(Efull_by_mask[mask][u]))
        out.append((t, agree, E_old, E_full, mask))
    return out


def pick(cands, lam, which):
    best = None
    for c in cands:
        s = c[1] - lam * (c[2] if which == "old" else c[3])
        if best is None or s > best[0]:
            best = (s, c)
    return best[1]


def route(H, taus):
    n = len(H[0]); u = np.full(n, 3); remaining = np.arange(n)
    for k in range(3):
        if taus[k] is None or len(remaining) == 0:
            continue
        take = H[k][remaining] < taus[k]; u[remaining[take]] = k; remaining = remaining[~take]
    return u


# ------------------------------------------------------------------------------------------------ the 2 x 2 study
def extract(fold, model, backbone: str, aug: str) -> dict:
    """Everything the cascade needs from one frozen backbone: exit costs, tap shapes, the clean and augmented exit
    heads, and the pooled features of the clean / augmented validation and test windows."""
    K = int(fold.n_classes); n = len(EXIT_TAPS[backbone]); input_shape = tuple(int(v) for v in fold.X_tr.shape[1:])
    costs = exit_costs(model, backbone, input_shape); full = float(costs["full_flops"]); ecost = [float(c) for c in costs["exit_flops"]] + [full]
    tapm = tap_model(model, backbone)
    tap = [{"shape": [int(v) for v in o.shape[1:]], "T": int(o.shape[1]), "per_step": int(np.prod(o.shape[2:]))} for o in tapm.outputs[:n]]
    (Ztr_a, _), (Zva_a, Pva_a) = views(tapm, n, make_augmenter(aug, fold, SEED), [fold.X_tr, fold.X_va])
    (Ztr_c, _), (Zva_c, Pva_c), (Zte_c, Pte_c) = (pooled_taps(tapm, Xs, n) for Xs in (fold.X_tr, fold.X_va, fold.X_te))
    [(Zte_a, Pte_a)] = views(tapm, n, make_augmenter(aug, fold, TEST_SEED), [fold.X_te])
    y_tr, y_va = fold.y_tr, fold.y_va
    heads = {"clean": [fit_exit_head(Ztr_c[k], y_tr, Zva_c[k], y_va, K, SEED) for k in range(n)],
             "aug": [fit_exit_head(Ztr_a[k], np.tile(y_tr, COPIES), Zva_a[k], np.tile(y_va, COPIES), K, SEED) for k in range(n)]}
    del Ztr_a, Ztr_c; gc.collect()
    fl = model.get_layer("softmax"); fin = (fl.kernel.numpy().astype(np.float64), fl.bias.numpy().astype(np.float64))
    f64 = lambda Z, P: [z.astype(np.float64) for z in Z] + [P.astype(np.float64)]
    F = {"va_clean": f64(Zva_c, Pva_c), "va_aug": f64(Zva_a, Pva_a), "te": f64(Zte_c, Pte_c), "te_aug": f64(Zte_a, Pte_a)}
    return {"K": K, "ecost": ecost, "full": full, "tap": tap, "y_te": fold.y_te, "n_te": int(len(fold.y_te)),
            "heads": {h: W + [fin] for h, W in heads.items()}, "F": F}


def cell(E: dict, hk: str) -> dict:
    """One (backbone, heads) cell: each exit alone, the label flips under the augmentation, and the cascade for every
    lambda with thresholds chosen on clean ("clean") or augmented ("aug") validation windows."""
    heads = Heads(E["heads"][hk]); K = E["K"]; F = E["F"]; y = E["y_te"]; n_te = E["n_te"]
    D = [int(F["te"][k].shape[1]) for k in range(3)]
    pool = [POOL_OPS_PER_ELEMENT * t["T"] * t["per_step"] for t in E["tap"]]
    assert all(3 * t["per_step"] == D[k] for k, t in enumerate(E["tap"])), (D, E["tap"])
    Efull = cost_vectors(E["ecost"], E["full"], D, K, pool, False)
    Eold = [E["ecost"][k] / E["full"] for k in range(4)]
    P = {name: [heads.prob(k, F[name][k]) for k in range(4)] for name in F}
    H = {name: [entropy(p) for p in Ps] for name, Ps in P.items()}
    A = {name: np.stack([p.argmax(1) for p in Ps]) for name, Ps in P.items()}
    tile = lambda v: np.tile(v, COPIES) # augmented copies are stacked copy by copy
    assert A["te_aug"].shape[1] == COPIES * n_te
    rec = {"exit_wf1": [wf1(y, A["te"][k]) for k in range(4)],
           "exit_label_flip": [float(np.mean(A["te_aug"][k] != tile(A["te"][k]))) for k in range(4)],
           "thresholds": {}}
    allm = (True, True, True); key = lambda m: allm
    for tk in ("clean", "aug"):
        va = "va_clean" if tk == "clean" else "va_aug"
        cands = search({allm: H[va]}, {allm: A[va]}, key, Eold, Efull, SHARE_GRID)
        rows = []
        for lam in LAMBDAS:
            taus, agree, E_old, E_full, mask = pick(cands, lam, "full")
            u = route(H["te"], taus); pred = A["te"][u, np.arange(n_te)]
            ur = route(H["te_aug"], taus); pr = A["te_aug"][ur, np.arange(len(ur))]
            share = (np.bincount(u, minlength=4) / n_te).tolist()
            rows.append({"lambda": lam, "wf1": wf1(y, pred), "acc": float(np.mean(pred == y)),
                         "mac_full": float(np.mean(Efull[mask][u])),
                         "flop_full": float(np.dot(share, cost_per_exit_flops(E["ecost"], E["full"], E["tap"], K, mask))),
                         "exit_share": share, "mask": list(mask), "taus": [None if t is None else float(t) for t in taus],
                         "routing_flip": float(np.mean(ur != tile(u))), "output_flip": float(np.mean(pr != tile(pred)))})
        rec["thresholds"][tk] = rows
    return rec
