# Data augmentation for early-exit human activity recognition

Training code for our paper on data augmentation for early-exit human activity recognition (HAR) on
microcontrollers. It trains three small HAR backbones with and without an augmentation, fits early exits on the frozen
backbones, and evaluates the early-exit pipeline with leave-one-subject-out cross-validation on four datasets.

- **Backbones:** TinierHAR, TCN-0.25 and DeepConvLSTM-0.25.
- **Augmentations:** rotation, time warping, magnitude warping, jitter + offset.
- **Datasets:** TORUS-HAR, PAMAP2, USC-HAD, DSADS (not included; see [datasets/README.md](datasets/README.md)).

## Setup

Python 3.11 and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The paper's runs used an NVIDIA GPU with TensorFlow 2.13.1, CUDA 11.8 and cuDNN 8.6. Each script prints `[GPU]` or
`[CPU]` for every fold; check for `[GPU]`, because TensorFlow falls back to the CPU silently when it cannot find CUDA.

## Usage

Two steps per dataset, backbone and augmentation. Each loops over all held-out subjects unless `--subjects` is given.

**1. Train the backbones.** `--aug none` trains the pure backbone; `rot`, `tw40`, `mw40` and `jo` train it with
rotation, time warping, magnitude warping or jitter + offset.

```bash
python train_backbone.py --dataset torus9 --backbone tinierhar --aug none --data-root /path/to/datasets
python train_backbone.py --dataset torus9 --backbone tinierhar --aug tw40 --data-root /path/to/datasets
```

**2. Fit the early exits and evaluate.** Needs both backbones from step 1. For the pure and the augmented backbone,
exit heads are fitted on clean and on augmented windows (the 2 x 2 of the paper), and the exit thresholds are chosen
for every cost weight lambda.

```bash
python train_exits.py --dataset torus9 --backbone tinierhar --aug tw40 --data-root /path/to/datasets
```

Options: `--dataset {torus9, pamap2_interp, usc_had, dsads}`, `--backbone {tinierhar, tcn_small, deepconvlstm_0_25}`,
`--subjects 540 11113 ...`, `--out outputs`, `--threads 2`. Finished folds are skipped, so an interrupted run can be
restarted with the same command.

## Folder layout

```
train_backbone.py      step 1: backbone training (one model per held-out subject)
train_exits.py         step 2: exit heads, thresholds and evaluation (the 2 x 2 study)
ee_har/
  config.py            datasets, training protocol, augmentation and early-exit settings
  data.py              raw loaders, windowing, leave-one-subject-out folds (split + standardisation)
  models.py            the three backbones and the layers their early exits tap
  augment.py           rotation, time warping, magnitude warping, jitter + offset
  training.py          determinism, pure training (Keras fit) and augmented training (custom train step)
  early_exit.py        exit costs, tap pooling, exit heads, threshold search, cascade evaluation
datasets/README.md     where to get the datasets and how to arrange them
```

## Outputs

Written under `outputs/` (one file per held-out subject):

- `models/<dataset>/<backbone>/<aug>/subj_<N>.weights.h5` and `.meta.json`: trained backbone and its scaler.
- `backbones/<dataset>/<backbone>/<aug>/subj_<N>.json`: weighted F1 of the backbone alone on the held-out subject.
- `exits/<dataset>/<backbone>/<aug>/subj_<N>.json`: the early-exit results. Under
  `backbones.{pure, aug}.cells.{clean, aug}` (backbone x exit heads) are the weighted F1 of each exit alone, the share
  of answers that change under the augmentation, and, per threshold set (`thresholds.clean` / `thresholds.aug`) and
  lambda, the cascade's weighted F1, its cost as a share of one full inference in FLOPs (`flop_full`) and MACs
  (`mac_full`), and the share of windows leaving at each exit. The paper reports lambda = 0.4 with `thresholds.clean`.

## Method in brief

- **Protocol:** leave one subject out; the remaining windows are split 80/20 (stratified, seed 42) into training and
  validation; per-channel standardisation fitted on the training windows. PAMAP2 subject 109 (24 windows) is never a
  test subject.
- **Backbone training:** Adam with a linear warm-up from 1e-6 to 1e-3 over 10 epochs and cosine decay to 1e-6 at
  epoch 150, gradient clipping at norm 3, batch 64, early stopping on the validation loss (patience 20). An
  augmentation is drawn afresh for every training batch.
- **Early exits:** three taps per backbone; each tap is pooled over time as [mean; max; std] and fed to one softmax
  layer, fitted on the frozen backbone (Adam 1e-3, up to 200 epochs, patience 10). Augmented heads use four augmented
  copies of every training and validation window. The fourth exit is the backbone's own classifier.
- **Stopping rule:** a window leaves at the first early exit whose softmax entropy is below that exit's threshold.
  Thresholds are chosen on validation windows to maximise agreement with the final exit minus lambda x the expected
  cost.

## Reproducibility

Runs are deterministic (`tf.config.experimental.enable_op_determinism`, fixed seeds): on the same GPU and software
versions, repeated runs give identical results. Other GPUs, the CPU or other library versions can give slightly
different numbers. With Keras 2.13, early stopping restores the best weights only when it fires; a run that reaches the
epoch limit keeps its final weights.

## Credits

- **TinierHAR:** our TensorFlow re-implementation based on the paper by Bian et al., "TinierHAR: Towards
  Ultra-Lightweight Deep Learning Models for Efficient Human Activity Recognition on Edge Devices", ISWC 2025. Original
  code: https://github.com/zhaxidele/TinierHAR.
- **DeepConvLSTM:** Ordóñez and Roggen, Sensors 16(1):115, 2016; the width-0.25 configuration follows the TinyHAR
  comparison code, https://github.com/teco-kit/ISWC22-HAR.
- **Time and magnitude warping:** re-implemented from Um et al., ICMI 2017,
  https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data.

## License

MIT; see [LICENSE](LICENSE).
