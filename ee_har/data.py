"""Raw loaders for the four datasets, windowing, and the leave-one-subject-out fold (split + standardisation).

The loader functions are copied unchanged from the code that produced the paper's results.  Window ORDER matters: the
validation split, the scaler and the training batches all depend on it, so the loaders keep the original
(lexicographic) file order.
"""
from __future__ import annotations

import bisect
import csv
import math
import os
from dataclasses import dataclass

import numpy as np
from scipy.io import loadmat

from .config import DATASETS, data_dir, window_samples

Segment = tuple[np.ndarray, int, int]   # (signal [T, C] float32, class index, subject id)

# =============================================================================
# Dataset container
# =============================================================================
@dataclass
class Dataset:
    """Windowed dataset in raw physical units (nothing standardised).

    X [N, T, C] float32; y [N] class index; subj [N] subject id; seg [N] id of the contiguous (subject, activity)
    segment each window came from (dense, in loader order, over the segments that yield windows)."""
    name: str
    X: np.ndarray
    y: np.ndarray
    subj: np.ndarray
    seg: np.ndarray
    class_names: list[str]
    fs: int
    channel_names: list[str]
    layout: list[dict]

# =============================================================================
# PAMAP2
# =============================================================================
PAMAP2_ACT_IDS = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]

PAMAP2_ACT_ID_TO_CLASS = {aid: i for i, aid in enumerate(PAMAP2_ACT_IDS)}

# 0-based columns of the +/-16g accelerometer (readme: "acceleration data (ms-2), scale: +/-16g, resolution 13-bit",
# columns 2-4 of each 17-column IMU block that starts at 3 (hand), 20 (chest), 37 (ankle)) and of the gyroscope
# (columns 8-10 of the block).
PAMAP2_INTERP_HAND = [4, 5, 6, 10, 11, 12]

PAMAP2_INTERP_CHEST = [21, 22, 23, 27, 28, 29]

PAMAP2_INTERP_ANKLE = [38, 39, 40, 44, 45, 46]

PAMAP2_INTERP_CHANNELS = PAMAP2_INTERP_HAND + PAMAP2_INTERP_CHEST + PAMAP2_INTERP_ANKLE

PAMAP2_MIN_SEGMENT_ROWS = 10

def interpolate_nan_runs(block: np.ndarray) -> np.ndarray:
    """Fill the NaN runs of every column of a [T, C] block by linear interpolation along the row index
    (``pandas.DataFrame.interpolate(method="linear", limit_direction="both")`` semantics: interior NaNs are
    interpolated linearly between the neighbouring valid samples, leading / trailing NaNs take the nearest valid
    sample, which is what ``numpy.interp`` does outside the range of its sample points).  A column without any valid
    sample is left as it is (NaN); the PAMAP2 Protocol blocks never contain one.  Returns a new float32 array."""
    out = np.array(block, dtype=np.float32, copy=True)
    idx = np.arange(out.shape[0])
    for c in range(out.shape[1]):
        valid = ~np.isnan(out[:, c])
        if valid.all() or not valid.any():
            continue
        out[:, c] = np.interp(idx, idx[valid], out[valid, c])
    return out

def load_pamap2_interp(data_dir: str) -> list[Segment]:
    """PAMAP2 Protocol files -> [(signal [T, 18], class, subject)].

    Channels: +/-16 g accelerometer xyz and gyroscope xyz of the hand, chest and ankle IMUs (file columns 4-6 / 10-12,
    21-23 / 27-29, 38-40 / 44-46), in that order; the 12 protocol activities 1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24.
    A segment is a maximal block of one activity; the wireless drop-outs (NaN runs, 1.1 % of the labelled rows) are
    filled by linear interpolation along time inside each block.  7611 windows of 5 s with 50 % overlap."""
    segments: list[Segment] = []
    for fname in sorted(os.listdir(data_dir)):
        if not (fname.startswith("subject") and fname.endswith(".dat")):
            continue
        subj_id = int(fname[len("subject"):-len(".dat")])
        arr = np.loadtxt(os.path.join(data_dir, fname))

        activity = arr[:, 1].astype(int)
        signal = arr[:, PAMAP2_INTERP_CHANNELS].astype(np.float32)

        i, n = 0, len(activity)
        while i < n:
            if activity[i] not in PAMAP2_ACT_ID_TO_CLASS: # 0 = transient, or an activity outside the protocol set
                i += 1
                continue
            j = i
            act = activity[i]
            while j < n and activity[j] == act:
                j += 1
            if j - i >= PAMAP2_MIN_SEGMENT_ROWS:
                block = interpolate_nan_runs(signal[i:j])
                if np.isnan(block).any():
                    raise ValueError(f"{fname}: rows {i}-{j} of activity {act} have a channel without any valid sample")
                segments.append((block, PAMAP2_ACT_ID_TO_CLASS[act], subj_id))
            i = j
    return segments

# =============================================================================
# USC-HAD
# =============================================================================
def load_usc_had(data_dir: str) -> list[Segment]:
    """USC-HAD -> [(signal [T, 6], class, subject)].

    One .mat per trial under Subject<N>/a<A>t<T>.mat; ``sensor_readings`` [T, 6] = acc xyz + gyro xyz at 100 Hz,
    ``activity_number`` 1..12 -> class 0..11.  Trials whose activity number is missing, whose readings are not [T, 6],
    whose activity is outside 1..12 or with fewer than 10 samples are skipped."""
    segments: list[Segment] = []
    subject_dirs = sorted(
        d for d in os.listdir(data_dir)
        if d.lower().startswith("subject") and os.path.isdir(os.path.join(data_dir, d))
    )
    for subj_dir in subject_dirs:
        subj_id = int(subj_dir[len("Subject"):])
        subj_path = os.path.join(data_dir, subj_dir)
        for fname in sorted(os.listdir(subj_path)):
            if not (fname.startswith("a") and fname.endswith(".mat")):
                continue
            m = loadmat(os.path.join(subj_path, fname))
            try:
                act_id = int(m["activity_number"][0])
            except (KeyError, ValueError):
                continue
            sig = np.asarray(m.get("sensor_readings"), dtype=np.float32)
            if sig.ndim != 2 or sig.shape[1] != 6:
                continue
            if act_id < 1 or act_id > 12:
                continue
            cls = act_id - 1
            if len(sig) >= 10:
                segments.append((sig, cls, subj_id))
    return segments

# =============================================================================
# DSADS
# =============================================================================
DSADS_N_ACTIVITIES = 19

DSADS_N_SUBJECTS = 8

DSADS_N_CLIPS = 60

DSADS_CLIP_SHAPE = (125, 45)

def load_dsads(data_dir: str) -> list[Segment]:
    """DSADS -> [(signal [T, 45], class, subject)].

    5 IMUs (torso, right arm, left arm, right leg, left leg) x 9 axes (acc, gyro, mag) at 25 Hz, 19 activities,
    8 subjects.  The 60 five-second clips of one (subject, activity) come from one continuous recording and are
    concatenated in clip order into one segment.  Segment order: activity (a01..a19), then subject (p1..p8)."""
    segments: list[Segment] = []
    for act_idx in range(DSADS_N_ACTIVITIES):
        act_folder = os.path.join(data_dir, f"a{act_idx + 1:02d}")
        if not os.path.isdir(act_folder):
            print(f"   WARNING: {act_folder} missing, skipping activity {act_idx + 1}")
            continue
        for subj_idx in range(DSADS_N_SUBJECTS):
            subj_folder = os.path.join(act_folder, f"p{subj_idx + 1}")
            if not os.path.isdir(subj_folder):
                continue
            seg_arrays = []
            for seg_idx in range(DSADS_N_CLIPS):
                seg_file = os.path.join(subj_folder, f"s{seg_idx + 1:02d}.txt")
                if not os.path.exists(seg_file):
                    continue
                arr = np.loadtxt(seg_file, delimiter=",", dtype=np.float32)
                if arr.shape != DSADS_CLIP_SHAPE:
                    print(f"   WARNING: {seg_file} has shape {arr.shape}, expected {DSADS_CLIP_SHAPE} -- skipping")
                    continue
                seg_arrays.append(arr)
            if not seg_arrays:
                continue
            full_signal = np.concatenate(seg_arrays, axis=0).astype(np.float32)
            segments.append((full_signal, act_idx, subj_idx + 1))
    return segments

# =============================================================================
# TORUS-HAR
# =============================================================================
# 9-class label set: stairs split normal (B1, B2) / assisted (B3 handrail, B4, B5 step-to) per direction, walking
# split by pace (A2 + C2 comfortable, A3 brisk).  Turning (A4) and the sit-to-stand trials (A6-*) are excluded.
TORUS9_LABELS = ["walking", "walking_brisk", "sitting", "standing", "lying",
                 "up_normal", "up_assist", "down_normal", "down_assist"]

TORUS9_LABEL_TO_CLASS = {n: i for i, n in enumerate(TORUS9_LABELS)}

TORUS9_CODE_TO_LABEL: dict[str, str] = {
    "A2": "walking", "C2": "walking",
    "A3": "walking_brisk",
    "A5": "sitting", "C3": "sitting",
    "A1": "standing", "C1": "standing", "D1": "standing",
    "C4": "lying",
    "B1_up": "up_normal", "B2_up": "up_normal",
    "B3_up": "up_assist", "B4_up": "up_assist", "B5_up": "up_assist",
    "B1_down": "down_normal", "B2_down": "down_normal",
    "B3_down": "down_assist", "B4_down": "down_assist", "B5_down": "down_assist",
}

TORUS_PLACEMENT = "wrist"

TORUS_CHANNEL_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "pressure"]

TORUS_GAP_MS = 1000.0 # packet-loss gaps longer than this split the segment

TORUS_PRESSURE_MA_SEC = 30

TORUS_MIN_SEGMENT_ROWS = 10

# Strings pandas.read_csv treats as missing by default (its na_values): NaN in the numeric columns, "no label" in the
# label column.
_PANDAS_NA = frozenset(["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
                        "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null"])

def _to_float(s: str) -> float:
    """One numeric CSV field with pandas' default missing-value strings -> float (NaN when missing)."""
    if s in _PANDAS_NA:
        return math.nan
    return float(s)

def _rolling_median_centred(x: np.ndarray, window: int) -> np.ndarray:
    """``pandas.Series(x).rolling(window, center=True, min_periods=1).median()`` without pandas.

    NaN values do not count; the median of an even count is the mean of the two middle values; an all-NaN window
    gives NaN."""
    vals = np.asarray(x, dtype=np.float64).tolist()
    n = len(vals)
    offset = (window - 1) // 2
    end_raw = np.arange(1 + offset, n + 1 + offset, dtype=np.int64)
    end = np.clip(end_raw, 0, n).tolist()
    start = np.clip(end_raw - window, 0, n).tolist()
    out = np.empty(n, dtype=np.float64)
    sorted_vals: list[float] = []
    prev_s = prev_e = 0
    for i in range(n):
        s, e = start[i], end[i]
        for j in range(prev_e, e):
            v = vals[j]
            if v == v:
                bisect.insort(sorted_vals, v)
        for j in range(prev_s, s):
            v = vals[j]
            if v == v:
                del sorted_vals[bisect.bisect_left(sorted_vals, v)]
        nobs = len(sorted_vals)
        if nobs >= 1:
            mid = nobs // 2
            out[i] = sorted_vals[mid] if nobs % 2 else (sorted_vals[mid] + sorted_vals[mid - 1]) / 2
        else:
            out[i] = math.nan
        prev_s, prev_e = s, e
    return out

def _read_torus_csv(path: str) -> tuple[np.ndarray, np.ndarray, list[str | None]]:
    """Read one TORUS-HAR wearable CSV -> (wearable_ts [n] float64, channels [n, 7] float64, label [n] str | None),
    equivalent to ``pd.read_csv(path, usecols=["wearable_ts", "label"] + TORUS_CHANNEL_COLS)``."""
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col = {name: header.index(name) for name in ["wearable_ts", "label"] + TORUS_CHANNEL_COLS}
        num_idx = [col["wearable_ts"]] + [col[c] for c in TORUS_CHANNEL_COLS]
        lab_idx = col["label"]
        numeric: list[list[float]] = []
        labels: list[str | None] = []
        for row in reader:
            if not row:
                continue
            numeric.append([_to_float(row[i]) for i in num_idx])
            lab = row[lab_idx]
            labels.append(None if lab in _PANDAS_NA else lab)
    num = np.asarray(numeric, dtype=np.float64).reshape(len(numeric), 1 + len(TORUS_CHANNEL_COLS))
    return num[:, 0], num[:, 1:], labels

def _load_torus_impl(data_dir: str, code_to_label: dict[str, str], label_to_class: dict[str, int],
                     fs: int) -> list[Segment]:
    """TORUS-HAR reader.

    Per subject folder (sorted): the wrist CSV is stable-sorted by wearable_ts, rows with a duplicated timestamp are
    dropped (first kept), the pressure channel is detrended on the whole session by subtracting its centred 30 s
    rolling median, the label codes are mapped to classes and unlabelled rows dropped.  Runs of one class are split
    where the clock gap exceeds TORUS_GAP_MS; runs with >= 10 rows are resampled onto a uniform fs grid."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"expected TORUS-HAR at {data_dir}: check --data-root / EE_HAR_DATA "
                                f"(see datasets/README.md)")
    segments: list[Segment] = []
    for folder in sorted(os.listdir(data_dir)):
        subj_dir = os.path.join(data_dir, folder)
        if not os.path.isdir(subj_dir):
            continue
        digits = "".join(c for c in folder if c.isdigit())
        if not digits:
            continue
        subj_id = int(digits)
        path = os.path.join(subj_dir, f"{folder}_{TORUS_PLACEMENT}.csv")
        if not os.path.exists(path):
            print(f"   WARNING: {path} missing, skipping subject {subj_id}")
            continue

        ts_all, sig_all, lab_all = _read_torus_csv(path)
        order = np.argsort(ts_all, kind="stable") # d.sort_values("wearable_ts", kind="stable")
        ts_all, sig_all = ts_all[order], sig_all[order]
        lab_all = [lab_all[k] for k in order.tolist()]
        first = np.ones(len(ts_all), dtype=bool) # d[~d["wearable_ts"].duplicated()]
        first[1:] = ~((ts_all[1:] == ts_all[:-1]) | (np.isnan(ts_all[1:]) & np.isnan(ts_all[:-1])))
        ts_all, sig_all = ts_all[first], sig_all[first]
        lab_all = [lab for lab, k in zip(lab_all, first.tolist()) if k]

        # Detrend pressure on the FULL continuous signal, before unlabelled rows are dropped.
        sig32 = sig_all.astype(np.float32)
        if TORUS_PRESSURE_MA_SEC:
            w = int(TORUS_PRESSURE_MA_SEC * fs)
            pr = sig_all[:, TORUS_CHANNEL_COLS.index("pressure")]
            base = _rolling_median_centred(pr, w)
            sig32[:, TORUS_CHANNEL_COLS.index("pressure")] = (pr - base).astype(np.float32)

        cls_all = [label_to_class.get(code_to_label.get(lab)) if lab is not None else None for lab in lab_all]
        keep = np.asarray([c is not None for c in cls_all], dtype=bool)
        if not keep.any():
            print(f"   WARNING: no labelled rows for subject {subj_id}")
            continue
        ts = ts_all[keep]
        sig = sig32[keep]
        cls = np.asarray([c for c in cls_all if c is not None], dtype=np.int32)

        brk = np.ones(len(ts), dtype=bool)
        brk[1:] = (np.diff(ts) > TORUS_GAP_MS) | (cls[1:] != cls[:-1])
        seg_id = np.cumsum(brk)
        for sid in np.unique(seg_id):
            m = seg_id == sid
            if m.sum() < TORUS_MIN_SEGMENT_ROWS:
                continue
            t_seg, s_seg = ts[m], sig[m]
            grid = np.arange(t_seg[0], t_seg[-1] + 1e-9, 1000.0 / fs)
            if len(grid) < TORUS_MIN_SEGMENT_ROWS:
                continue
            res = np.empty((len(grid), s_seg.shape[1]), dtype=np.float32)
            for c in range(s_seg.shape[1]):
                res[:, c] = np.interp(grid, t_seg, s_seg[:, c])
            segments.append((res, int(cls[m][0]), subj_id))
    return segments

def load_torus9(data_dir: str, fs: int) -> list[Segment]:
    """TORUS-HAR, 9-class label set -> [(signal [T, 7], class, subject)]."""
    return _load_torus_impl(data_dir, TORUS9_CODE_TO_LABEL, TORUS9_LABEL_TO_CLASS, fs)

# =============================================================================
# Windowing
# =============================================================================
def sliding_window(segments: list[Segment], win: int, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slice each segment into windows of ``win`` samples with stride ``step``.  Segments shorter than ``win`` yield
    nothing and consume no segment id.  Returns X [N, win, C] float32, y, subj, seg [N] int64; windows are ordered by
    segment, then by start position."""
    X, y, subj, seg = [], [], [], []
    k = 0
    for sig, lab, s_id in segments:
        if len(sig) < win:
            continue
        for start in range(0, len(sig) - win + 1, step):
            X.append(sig[start:start + win])
            y.append(lab)
            subj.append(s_id)
            seg.append(k)
        k += 1
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64),
            np.asarray(subj, dtype=np.int64), np.asarray(seg, dtype=np.int64))


LOADERS = {
    "torus9": load_torus9,
    "pamap2_interp": lambda d, fs: load_pamap2_interp(d),
    "usc_had": lambda d, fs: load_usc_had(d),
    "dsads": lambda d, fs: load_dsads(d),
}


def load_dataset(name: str, data_root: str | None = None) -> Dataset:
    """Load and window one dataset; ``data_root`` is the folder holding the raw datasets (see datasets/README.md)."""
    cfg = DATASETS[name]
    win, step = window_samples(name)
    path = data_dir(name, data_root)
    if not os.path.isdir(path):
        raise SystemExit(f"{name}: no data at {path}; check --data-root / EE_HAR_DATA (see datasets/README.md)")
    segments = LOADERS[name](path, int(cfg["fs"]))
    X, y, subj, seg = sliding_window(segments, win, step)
    if X.ndim != 3 or X.shape[2] != cfg["n_channels"]:
        raise ValueError(f"{name}: windows have shape {X.shape}, expected [N, {win}, {cfg['n_channels']}]")
    return Dataset(name=name, X=X, y=y, subj=subj, seg=seg, class_names=list(cfg["class_names"]), fs=int(cfg["fs"]),
                   channel_names=list(cfg["channel_names"]), layout=list(cfg["layout"]))


@dataclass
class Fold:
    """One leave-one-subject-out fold: X_* are standardised with the scaler fitted on the training windows only."""
    dataset: str
    subject: int
    seed: int
    mean: np.ndarray
    std: np.ndarray
    X_tr: np.ndarray
    X_va: np.ndarray
    X_te: np.ndarray
    y_tr: np.ndarray
    y_va: np.ndarray
    y_te: np.ndarray
    n_classes: int
    fs: float
    layout: list
    seg_te: np.ndarray


def make_fold(ds: Dataset, subject: int, seed: int) -> Fold:
    """Test = the held-out subject; the other subjects' windows are split 80/20 (stratified, random_state = seed) into
    training and validation; per-channel standardisation fitted on the training windows."""
    from .training import apply_scaler, fit_scaler, loso_split
    tr_idx, va_idx, te_idx = loso_split(ds.y, ds.subj, subject, seed)
    mean, std = fit_scaler(ds.X[tr_idx])
    return Fold(dataset=ds.name, subject=int(subject), seed=int(seed), mean=np.asarray(mean), std=np.asarray(std),
                X_tr=np.asarray(apply_scaler(ds.X[tr_idx], mean, std), np.float32),
                X_va=np.asarray(apply_scaler(ds.X[va_idx], mean, std), np.float32),
                X_te=np.asarray(apply_scaler(ds.X[te_idx], mean, std), np.float32),
                y_tr=ds.y[tr_idx], y_va=ds.y[va_idx], y_te=ds.y[te_idx], n_classes=len(ds.class_names), fs=float(ds.fs),
                layout=list(ds.layout), seg_te=ds.seg[te_idx])
