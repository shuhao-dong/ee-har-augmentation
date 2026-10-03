"""The four augmentations.  Each factory returns ``augment(x_std) -> x_std_aug`` for a batch of fold-standardised
windows [B, T, C]; every augmenter owns a ``tf.random.Generator.from_seed(seed)`` and draws afresh on every call.

* rotation: heading U(-pi, pi) about the window's gravity direction and tilt U(-15, 15) degrees about a random
  perpendicular axis; the same rotation for the accelerometer, gyroscope and magnetometer of one IMU.
* time warping and magnitude warping: re-implemented in TensorFlow from Um et al. (ICMI 2017),
  https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data.  Time warping uses one clock per window for
  all channels; magnitude warping warps the movement only (gravity, a 0.3 Hz low-pass, is kept) with one curve per IMU.
* jitter + offset: white noise (0.05 channel SD) plus a constant offset per window and channel (0.05 channel SD) on
  the IMU channels.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import tensorflow as tf

def make_rotation_augmenter(layout: Sequence[dict], scaler_mean: np.ndarray, scaler_std: np.ndarray,
                            max_tilt_deg: float = 15.0, seed: int = 42) -> Callable[[tf.Tensor], tf.Tensor]:
    """Return ``augment(x_std) -> x_std_aug`` for fold-standardised batches [B, T, C].

    ``layout`` is the per-IMU list of ``{"acc": [i, j, k], "gyro": [...], "mag": [...] | None}`` channel triads.  Every
    triad of one IMU is rotated by the same random rotation (heading about the window's mean acceleration direction, then
    a tilt about a random perpendicular axis), drawn from ``tf.random.Generator.from_seed(seed)`` per IMU and per batch."""
    mu = tf.constant(np.asarray(scaler_mean, np.float32))
    sigma = tf.constant(np.asarray(scaler_std, np.float32))
    sens = [{k: [int(i) for i in s[k]] for k in ("acc", "gyro", "mag") if s.get(k) is not None} for s in layout]
    C = int(mu.shape[0])
    g = tf.random.Generator.from_seed(int(seed))
    max_tilt = float(np.deg2rad(max_tilt_deg))

    def rodrigues(axis: tf.Tensor, angle: tf.Tensor) -> tf.Tensor:
        """axis [B, 3] unit, angle [B] -> R [B, 3, 3]."""
        ax, ay, az = axis[:, 0], axis[:, 1], axis[:, 2]
        zero = tf.zeros_like(ax)
        K = tf.stack([tf.stack([zero, -az, ay], 1), tf.stack([az, zero, -ax], 1), tf.stack([-ay, ax, zero], 1)], 1)
        I = tf.eye(3, batch_shape=[tf.shape(axis)[0]])
        s = tf.sin(angle)[:, None, None]
        c = tf.cos(angle)[:, None, None]
        return I + s * K + (1.0 - c) * tf.matmul(K, K)

    def augment(x_std: tf.Tensor) -> tf.Tensor:
        """Rotate the IMU triads of a standardised batch (in raw units; the fold scaler is undone and re-applied)."""
        B = tf.shape(x_std)[0]
        x = x_std * sigma + mu # raw units [B, T, C]
        cols = [x[:, :, i] for i in range(C)]
        for s in sens:
            acc = tf.stack([cols[i] for i in s["acc"]], axis=2) # [B, T, 3]
            grav = tf.reduce_mean(acc, axis=1)
            grav = grav / (tf.norm(grav, axis=1, keepdims=True) + 1e-8)
            theta = g.uniform([B], -np.pi, np.pi)
            R_head = rodrigues(grav, theta)
            rnd = g.normal([B, 3])
            perp = rnd - tf.reduce_sum(rnd * grav, 1, keepdims=True) * grav
            perp = perp / (tf.norm(perp, axis=1, keepdims=True) + 1e-8)
            phi = g.uniform([B], -max_tilt, max_tilt)
            R = tf.matmul(rodrigues(perp, phi), R_head) # [B, 3, 3]
            for key in ("acc", "gyro", "mag"):
                if key in s:
                    v = tf.stack([cols[i] for i in s[key]], axis=2) # [B, T, 3]
                    vr = tf.einsum("bij,btj->bti", R, v)
                    for j, i in enumerate(s[key]):
                        cols[i] = vr[:, :, j]
        x_rot = tf.stack(cols, axis=2)
        return (x_rot - mu) / sigma

    return augment


SIGMA = 0.2

KNOT = 4

FLOOR = 0.05

def spline_basis(T: int, knot: int = KNOT) -> np.ndarray:
    """[T, knot + 2] matrix B with curve = B @ y for knot values y at np.linspace(0, T - 1, knot + 2) (the released
    code's knot positions), cubic spline with scipy's default not-a-knot ends: the spline is linear in y."""
    from scipy.interpolate import CubicSpline
    xx = np.linspace(0.0, T - 1.0, knot + 2); t = np.arange(T, dtype=np.float64)
    return np.stack([CubicSpline(xx, np.eye(knot + 2)[j])(t) for j in range(knot + 2)], 1)

def make_timewarp_augmenter(layout: Sequence[dict], scaler_mean: np.ndarray, scaler_std: np.ndarray,
                            sigma: float = SIGMA, knot: int = KNOT, seed: int = 42) -> Callable:
    """``augment(x_std)``; layout / scaler arguments are accepted for signature compatibility with the rotation
    augmenter and unused (the warp is per window, all channels, scale-free)."""
    import tensorflow as tf
    g = tf.random.Generator.from_seed(int(seed))
    cache = {}

    def augment(x_std: tf.Tensor) -> tf.Tensor:
        x = tf.convert_to_tensor(x_std, tf.float32)
        T = int(x.shape[1]) if x.shape[1] is not None else None
        if T not in cache:                                                             
            cache[T] = spline_basis(T, knot).T.astype(np.float32)                     
        Bm = tf.constant(cache[T])                                                     
        B = tf.shape(x)[0]
        yy = g.normal([B, knot + 2], mean=1.0, stddev=float(sigma))
        step = tf.maximum(tf.matmul(yy, Bm), FLOOR) # [B, T] local time step
        cum = tf.concat([tf.zeros([B, 1]), tf.cumsum(step[:, :-1], axis=1)], 1)        
        tt = cum * ((T - 1.0) / cum[:, -1:])                                           
        grid = tf.tile(tf.range(T, dtype=tf.float32)[None, :], [B, 1]) # [B, T] output times
        j = tf.clip_by_value(tf.searchsorted(tt, grid, side="right") - 1, 0, T - 2) # tt[j] <= t < tt[j+1]
        t0 = tf.gather(tt, j, batch_dims=1); t1 = tf.gather(tt, j + 1, batch_dims=1)
        w = tf.clip_by_value((grid - t0) / (t1 - t0), 0.0, 1.0)[:, :, None]
        x0 = tf.gather(x, j, batch_dims=1); x1 = tf.gather(x, j + 1, batch_dims=1)
        return (1.0 - w) * x0 + w * x1

    return augment


MW_SIGMA = 0.2

MW_KNOT = 4

MW_FLOOR = 0.05

GRAVITY_CUTOFF_HZ = 0.3

JIT_SD = 0.05

OFF_SD = 0.05

def lowpass_operator(T: int, fs: float, cutoff: float = GRAVITY_CUTOFF_HZ) -> np.ndarray:
    """[T, T] matrix L with filtfilt(b, a, x) = L @ x (2nd-order Butterworth low-pass at ``cutoff`` Hz, zero phase).
    Even (mirror) padding of T - 1 samples keeps the estimate from sticking to the edge samples; the DC gain is 1."""
    from scipy.signal import butter, filtfilt
    b, a = butter(2, cutoff / (fs / 2.0), "low")
    return filtfilt(b, a, np.eye(T), axis=0, padtype="even", padlen=T - 1)

def _imu_channels(layout: Sequence[dict], keys=("acc", "gyro", "mag")) -> list[int]:
    return sorted(int(i) for s in layout for k in keys if s.get(k) is not None for i in s[k])

def make_magwarp_augmenter(layout: Sequence[dict], scaler_mean: np.ndarray, scaler_std: np.ndarray, fs: float,
                           sigma: float = MW_SIGMA, knot: int = MW_KNOT, seed: int = 42) -> Callable:
    import tensorflow as tf
    mu_np = np.asarray(scaler_mean, np.float32); sd_np = np.asarray(scaler_std, np.float32)
    sens = [{k: [int(i) for i in s[k]] for k in ("acc", "gyro") if s.get(k) is not None} for s in layout]
    g = tf.random.Generator.from_seed(int(seed))
    cache = {}

    def augment(x_std: tf.Tensor) -> tf.Tensor:
        x_std = tf.convert_to_tensor(x_std, tf.float32)
        T = int(x_std.shape[1]); C = int(x_std.shape[2])
        if T not in cache:                                      
            cache[T] = (spline_basis(T, knot).T.astype(np.float32), lowpass_operator(T, fs).astype(np.float32))
        Bm = tf.constant(cache[T][0]); Lp = tf.constant(cache[T][1])
        mu = tf.constant(mu_np); sd = tf.constant(sd_np)
        x = x_std * sd + mu                                      
        B = tf.shape(x)[0]
        cols = [x[:, :, i] for i in range(C)]
        for s in sens:
            c = tf.maximum(tf.matmul(g.normal([B, knot + 2], mean=1.0, stddev=float(sigma)), Bm), MW_FLOOR) # [B, T]
            if "acc" in s:
                a = tf.stack([cols[i] for i in s["acc"]], axis=2)# [B, T, 3]
                grav = tf.einsum("ts,bsc->btc", Lp, a)
                w = grav + c[:, :, None] * (a - grav)
                for j, i in enumerate(s["acc"]):
                    cols[i] = w[:, :, j]
            if "gyro" in s:
                for i in s["gyro"]:
                    cols[i] = c * cols[i]
        return (tf.stack(cols, axis=2) - mu) / sd

    return augment

def make_jitter_offset_augmenter(layout: Sequence[dict], scaler_mean=None, scaler_std=None, jitter_sd: float = JIT_SD,
                                 offset_sd: float = OFF_SD, seed: int = 42) -> Callable:
    import tensorflow as tf
    chans = _imu_channels(layout)
    g = tf.random.Generator.from_seed(int(seed))

    def augment(x_std: tf.Tensor) -> tf.Tensor:
        x = tf.convert_to_tensor(x_std, tf.float32)
        C = int(x.shape[2])
        m = np.zeros(C, np.float32); m[chans] = 1.0; m = tf.constant(m) # IMU channels only
        noise = g.normal(tf.shape(x), stddev=float(jitter_sd))
        off = g.normal([tf.shape(x)[0], 1, C], stddev=float(offset_sd))
        return x + (noise + off) * m

    return augment


def make_augmenter(name: str, fold, seed: int) -> Callable:
    """The augmenter of the paper's setting ``name`` (config.AUGMENTATIONS) for this fold."""
    from .config import AUGMENTATIONS
    p = AUGMENTATIONS[name]
    if name == "rot":
        return make_rotation_augmenter(fold.layout, fold.mean, fold.std, max_tilt_deg=p["max_tilt_deg"], seed=seed)
    if name == "tw40":
        return make_timewarp_augmenter(fold.layout, fold.mean, fold.std, sigma=p["sigma"], seed=seed)
    if name == "mw40":
        return make_magwarp_augmenter(fold.layout, fold.mean, fold.std, fs=float(fold.fs), sigma=p["sigma"], seed=seed)
    if name == "jo":
        return make_jitter_offset_augmenter(fold.layout, seed=seed)
    raise ValueError(name)
