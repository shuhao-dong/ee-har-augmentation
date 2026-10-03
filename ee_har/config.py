"""Datasets, training protocol and the settings of the four augmentations and the early-exit pipeline."""
from __future__ import annotations

import os

# Folder of each dataset below the data root (see datasets/README.md).  The root comes from --data-root or the
# environment variable EE_HAR_DATA.
DATA_SUBDIR = {
    "torus9": "TORUS_HAR",
    "pamap2_interp": os.path.join("PAMAP2_Dataset", "Protocol"),
    "usc_had": "USC-HAD",
    "dsads": "DSADS",
}


def data_dir(name: str, data_root: str | None = None) -> str:
    root = data_root or os.environ.get("EE_HAR_DATA")
    if not root:
        raise SystemExit("set --data-root or the environment variable EE_HAR_DATA (see datasets/README.md)")
    return os.path.join(root, DATA_SUBDIR[name])


def _imu_names(units: list[str], triads: tuple[str, ...]) -> list[str]:
    return [f"{u}_{t}_{ax}" for u in units for t in triads for ax in "xyz"]


# layout = the channel indices of every IMU's accelerometer, gyroscope and magnetometer triads
DATASETS: dict[str, dict] = {
    "torus9": {
        "fs": 100, "window_seconds": 2.5, "step_seconds": 1.25, "n_channels": 7, "n_classes": 9,
        "class_names": ["walking", "walking_brisk", "sitting", "standing", "lying",
                        "up_normal", "up_assist", "down_normal", "down_assist"],
        "channel_names": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "pressure"],
        "layout": [{"acc": [0, 1, 2], "gyro": [3, 4, 5], "mag": None}],
    },
    "pamap2_interp": {   # PAMAP2: +-16 g accelerometer and gyroscope of the hand, chest and ankle IMUs, drop-outs interpolated
        "fs": 100, "window_seconds": 5.0, "step_seconds": 2.5, "n_channels": 18, "n_classes": 12,
        "class_names": ["lying", "sitting", "standing", "walking", "running", "cycling",
                        "nordic_walk", "asc_stairs", "desc_stairs", "vacuum", "ironing", "rope_jump"],
        "channel_names": _imu_names(["hand", "chest", "ankle"], ("acc16g", "gyro")),
        "layout": [{"acc": [6 * k, 6 * k + 1, 6 * k + 2], "gyro": [6 * k + 3, 6 * k + 4, 6 * k + 5], "mag": None}
                   for k in range(3)],
    },
    "usc_had": {
        "fs": 100, "window_seconds": 5.0, "step_seconds": 2.5, "n_channels": 6, "n_classes": 12,
        "class_names": ["walk_fwd", "walk_left", "walk_right", "walk_up", "walk_down",
                        "run_fwd", "jump_up", "sitting", "standing", "sleeping",
                        "elevator_up", "elevator_down"],
        "channel_names": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        "layout": [{"acc": [0, 1, 2], "gyro": [3, 4, 5], "mag": None}],
    },
    "dsads": {
        "fs": 25, "window_seconds": 5.0, "step_seconds": 2.5, "n_channels": 45, "n_classes": 19,
        "class_names": ["sitting", "standing", "lying_back", "lying_right",
                        "asc_stairs", "desc_stairs", "elev_still", "elev_moving",
                        "walk_park", "walk_tread_f", "walk_tread_i",
                        "run_tread", "stepper", "cross_train",
                        "cycle_horiz", "cycle_vert", "rowing", "jumping", "basketball"],
        "channel_names": _imu_names(["T", "RA", "LA", "RL", "LL"], ("acc", "gyro", "mag")),
        "layout": [{"acc": [9 * k, 9 * k + 1, 9 * k + 2], "gyro": [9 * k + 3, 9 * k + 4, 9 * k + 5],
                    "mag": [9 * k + 6, 9 * k + 7, 9 * k + 8]} for k in range(5)],
    },
}


def window_samples(name: str) -> tuple[int, int]:
    """(window, stride) in samples: int(fs * seconds) (DSADS: 125 / 62)."""
    cfg = DATASETS[name]
    return int(cfg["fs"] * cfg["window_seconds"]), int(cfg["fs"] * cfg["step_seconds"])


# Backbone training (leave-one-subject-out; PAMAP2 subject 109, 24 windows of one class, is never a test subject
# but stays in the other folds' training data).
PROTOCOL: dict = dict(seed=42, val_fraction=0.2, epochs=150, patience=20, warmup_epochs=10, lr_min=1e-6, lr_max=1e-3,
                      batch_size=64, clip_norm=3.0, exclude_test_subjects={"pamap2_interp": [109]})
DROPOUT: dict = {"tinierhar": 0.3, "tcn_small": 0.1, "deepconvlstm_0_25": 0.5}
BACKBONES = ("tinierhar", "tcn_small", "deepconvlstm_0_25")

# The paper's four augmentations ("none" = the pure backbone, trained without augmentation)
AUGMENTATIONS: dict = {
    "rot": {"name": "rotation", "max_tilt_deg": 15.0, "heading": "U(-pi, pi)"},
    "tw40": {"name": "time warping", "sigma": 0.4, "knots": 4, "floor": 0.05},
    "mw40": {"name": "magnitude warping (gravity-preserving)", "sigma": 0.4, "knots": 4, "floor": 0.05,
             "gravity_lowpass_hz": 0.3},
    "jo": {"name": "jitter + offset", "jitter_sd": 0.05, "offset_sd": 0.05},
}

# Early exits: one softmax layer per tap on [mean; max; std] over time, fitted on the frozen backbone
EXIT_HEAD: dict = dict(ic_epochs=200, ic_patience=10, ic_lr=1e-3)
STUDY: dict = dict(copies=4, # augmented copies of every training / validation / test window
                   seed=42, test_seed=1042, # augmenter seeds for the head-fitting copies and for the test copies
                   lambdas=[0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8], # cost weight of the threshold objective
                   share_grid=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], # candidate exit shares per early exit
                   pool_ops_per_element=4, deployed_lambda=0.4)
