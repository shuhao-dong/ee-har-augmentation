"""Backbone training: determinism, the leave-one-subject-out split and scaler, and the two training loops.

* fit_supervised -- the pure backbones: Keras fit with sparse cross-entropy.
* fit_augmented  -- every augmented backbone: the same optimiser, schedule and early stopping, with the augmentation
  applied to each training batch inside a custom train step and a seeded tf.data pipeline.

Both: Adam (learning rate warmed up linearly from 1e-6 to 1e-3 over 10 epochs, then cosine decay to 1e-6 at epoch 150),
gradient clipping at norm 3, batch 64, early stopping on the validation loss with patience 20 (counted after the
warm-up).  Note that Keras 2.13 restores the best weights only when early stopping fires; a run that reaches 150
epochs keeps its final weights.
"""
from __future__ import annotations

import os
import time
from typing import Callable

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from .config import PROTOCOL

BATCH_SIZE: int = int(PROTOCOL["batch_size"])
EPOCHS: int = int(PROTOCOL["epochs"])
PATIENCE: int = int(PROTOCOL["patience"])
WARMUP: int = int(PROTOCOL["warmup_epochs"])
LR_MIN: float = float(PROTOCOL["lr_min"])
LR_MAX: float = float(PROTOCOL["lr_max"])
CLIP_NORM: float = float(PROTOCOL["clip_norm"])
VAL_FRACTION: float = float(PROTOCOL["val_fraction"])


def set_determinism(seed: int) -> None:
    """Process-level determinism: TF op determinism, GPU memory growth and keras.utils.set_random_seed(seed)."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    tf.config.experimental.enable_op_determinism()
    try:
        for gpu in tf.config.experimental.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except (ValueError, RuntimeError):
        pass
    keras.utils.set_random_seed(int(seed))

def loso_split(y: np.ndarray, subj: np.ndarray, test_subject: int,
               seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(tr_idx, va_idx, te_idx): the held-out subject is the test set; the remaining windows are split 80/20,
    stratified on the label, with ``random_state=seed``."""
    test_mask = subj == test_subject
    pool_idx = np.where(~test_mask)[0]
    te_idx = np.where(test_mask)[0]
    assert te_idx.size > 0 and test_subject not in np.unique(subj[pool_idx]), "LEAK: test subject in the pool"
    tr_idx, va_idx = train_test_split(pool_idx, test_size=VAL_FRACTION, stratify=y[pool_idx], random_state=seed)
    assert test_subject not in np.unique(subj[tr_idx]), "LEAK: test subject in train"
    assert test_subject not in np.unique(subj[va_idx]), "LEAK: test subject in val"
    return tr_idx, va_idx, te_idx

def fit_scaler(X_tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel (mean [C], std [C]) of a StandardScaler fitted on the training windows only."""
    sc = StandardScaler()
    sc.fit(X_tr.reshape(-1, X_tr.shape[-1]))
    return sc.mean_, sc.scale_

def apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardise [N, T, C] with the fold scaler; float32 output (bitwise ``StandardScaler.transform``)."""
    C = X.shape[-1]
    Z = np.array(X, dtype=(X.dtype if X.dtype.kind == "f" else np.float64), copy=True).reshape(-1, C)
    Z -= np.asarray(mean).astype(Z.dtype)
    Z /= np.asarray(std).astype(Z.dtype)
    return Z.reshape(X.shape).astype(np.float32)

class CosineWarmup(keras.callbacks.Callback):
    """Linear warm-up lr_min -> lr_max over ``warmup_epochs`` (reaching lr_max at the last warm-up epoch), then a
    cosine decay to lr_min at ``total_epochs``."""

    def __init__(self, lr_max: float, lr_min: float, warmup_epochs: int, total_epochs: int) -> None:
        super().__init__()
        self.lr_max, self.lr_min = lr_max, lr_min
        self.warmup, self.total = warmup_epochs, total_epochs

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        if epoch < self.warmup:
            lr = self.lr_min + (self.lr_max - self.lr_min) * (epoch + 1) / self.warmup
        else:
            t = epoch - self.warmup
            T = max(1, self.total - self.warmup)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + np.cos(np.pi * t / T))
        self.model.optimizer.learning_rate.assign(float(lr))

def early_stopping(patience: int = PATIENCE, start_from_epoch: int = WARMUP) -> keras.callbacks.EarlyStopping:
    """EarlyStopping on val_loss with restore_best_weights; the warm-up epochs do not count toward patience.
    Keras 2.13 restores the best weights only when the stop fires."""
    return keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True,
                                         verbose=0, start_from_epoch=start_from_epoch)

def fit_supervised(model: keras.Model, X_tr, y_tr: np.ndarray, X_va, y_va: np.ndarray, seed: int,
                   epochs: int = EPOCHS, patience: int = PATIENCE, start_from_epoch: int = WARMUP,
                   predict_batch_size: int = BATCH_SIZE) -> dict:
    """Sparse cross-entropy fit with Adam(lr_min, clipnorm) + CosineWarmup + EarlyStopping, batch 64 (the pure
    backbones); returns ``{"epochs_run", "val_f1", "seconds", "seed"}``.  The caller seeds with
    keras.utils.set_random_seed(seed) immediately before building ``model`` (models.build_backbone)."""
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR_MIN, clipnorm=CLIP_NORM),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [CosineWarmup(LR_MAX, LR_MIN, WARMUP, epochs),
                 early_stopping(patience=patience, start_from_epoch=start_from_epoch)]
    t0 = time.time()
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs, batch_size=BATCH_SIZE,
                     callbacks=callbacks, verbose=0)
    seconds = time.time() - t0
    y_va_pred = model.predict(X_va, batch_size=predict_batch_size, verbose=0).argmax(axis=1)
    return {"epochs_run": int(len(hist.history["loss"])),
            "val_f1": float(f1_score(y_va, y_va_pred, average="weighted")),
            "seconds": float(seconds), "seed": int(seed)}


class AugmentedTrainer(keras.Model):
    """Wraps a backbone for training with an augmentation: every training batch is replaced by ``augment(x)``; the
    loss is the batch-mean cross-entropy of the backbone's softmax; each gradient is clipped to norm 3 per variable
    before the optimiser (whose own clipnorm clips once more).  Validation batches are not augmented."""

    def __init__(self, backbone: keras.Model, augment: Callable[[tf.Tensor], tf.Tensor] | None,
                 clip_norm: float = CLIP_NORM, **kw) -> None:
        super().__init__(**kw)
        self.backbone = backbone
        self.augment = augment
        self.clip_norm = float(clip_norm)
        self._final_dense = backbone.get_layer("softmax")
        self.fwd_model = keras.Model(backbone.input, self._final_dense.input, name="backbone_features")
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    def call(self, x, training: bool = False):
        return self.backbone(x, training=training)

    def _probs(self, x, training: bool):
        features = self.fwd_model(x, training=training)
        if self._final_dense.use_bias:
            logits = tf.matmul(features, self._final_dense.kernel) + self._final_dense.bias
        else:
            logits = tf.matmul(features, self._final_dense.kernel)
        return tf.nn.softmax(logits, axis=-1)

    def _loss(self, x, y_true, training: bool):
        probs = self._probs(x, training)
        return tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(y_true, probs, from_logits=False)), probs

    def train_step(self, data):
        x, y_true = data
        if self.augment is not None:
            x = self.augment(x)
        with tf.GradientTape() as tape:
            loss, probs = self._loss(x, y_true, training=True)
        variables = list(self.backbone.trainable_variables)
        gradients = tape.gradient(loss, variables)
        gradients = [tf.clip_by_norm(g, self.clip_norm) if g is not None else None for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, variables))
        return self._track(loss, y_true, probs)

    def test_step(self, data):
        x, y_true = data
        loss, probs = self._loss(x, y_true, training=False)
        return self._track(loss, y_true, probs)

    def _track(self, loss, y_true, probs) -> dict:
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(y_true, probs)
        return {"loss": self.loss_tracker.result(), "accuracy": self.acc_tracker.result()}

    @property
    def metrics(self) -> list:
        return [self.loss_tracker, self.acc_tracker]


def fit_augmented(backbone: keras.Model, X_tr, y_tr, X_va, y_va, augment: Callable[[tf.Tensor], tf.Tensor] | None,
                  seed: int, epochs: int = EPOCHS, patience: int = PATIENCE) -> dict:
    """Train ``backbone`` in place with ``augment`` applied to every training batch; returns
    ``{"epochs_run", "val_f1", "seconds"}``.  Training batches come from
    ``from_tensor_slices(...).shuffle(N, seed=seed).batch(64)``, reshuffled every epoch."""
    X_tr = np.asarray(X_tr, np.float32)
    X_va = np.asarray(X_va, np.float32)
    y_tr = np.asarray(y_tr).astype(np.int32)
    y_va = np.asarray(y_va).astype(np.int32)
    model = AugmentedTrainer(backbone, augment)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR_MIN, clipnorm=CLIP_NORM))
    callbacks = [CosineWarmup(LR_MAX, LR_MIN, WARMUP, epochs), early_stopping(patience=patience, start_from_epoch=WARMUP)]
    train_ds = (tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
                .shuffle(len(X_tr), seed=int(seed)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
    val_ds = tf.data.Dataset.from_tensor_slices((X_va, y_va)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    t0 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=0)
    seconds = time.time() - t0
    y_va_pred = backbone.predict(X_va, batch_size=BATCH_SIZE, verbose=0).argmax(axis=1)
    return {"epochs_run": int(len(hist.history["loss"])),
            "val_f1": float(f1_score(y_va, y_va_pred, average="weighted")), "seconds": float(seconds)}
