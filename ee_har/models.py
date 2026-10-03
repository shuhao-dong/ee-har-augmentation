"""The three backbones: TinierHAR, TCN-0.25 and DeepConvLSTM-0.25 (Keras), and the layers their early exits tap.

* TinierHAR: our TensorFlow re-implementation based on the paper by Bian et al. (ISWC 2025); original PyTorch code at
  https://github.com/zhaxidele/TinierHAR.
* DeepConvLSTM (Ordonez and Roggen, Sensors 2016) at width 0.25, configured as in the TinyHAR comparison code
  (https://github.com/teco-kit/ISWC22-HAR).
* TCN-0.25: our small temporal convolutional network.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------
TCN_FILTERS = 16

TCN_DILATIONS = (1, 2, 4, 8)

TCN_KERNEL = 3

TCN_DROPOUT = 0.1

TINIERHAR_NB_CONV_BLOCKS = 4

TINIERHAR_NB_FILTERS = 4

TINIERHAR_NB_UNITS_GRU = 16

TINIERHAR_KERNEL = 5

TINIERHAR_RELEASED_DROP_PROB = 0.3

TORCH_BN_MOMENTUM = 0.9

TORCH_BN_EPSILON = 1e-5

def _shape2(input_shape: Sequence[int]) -> tuple[int, int]:
    """(T, C) as plain ints."""
    return int(input_shape[0]), int(input_shape[1])

# ----------------------------------------------------------------------------------------------------------------
# 1. TCN-0.25 (tcn_small)
# ----------------------------------------------------------------------------------------------------------------
def dilated_residual_block(x: tf.Tensor, filters: int, kernel_size: int, dilation: int, dropout: float,
                           idx: int) -> tf.Tensor:
    """Residual block of two dilated Conv1D-BN-ReLU-Dropout stages with a 1x1 skip conv on channel mismatch; the Add
    layer is named ``tcn_b{idx}_add`` (an early-exit tap)."""
    skip = x
    x = layers.Conv1D(filters, kernel_size, padding="same", dilation_rate=dilation, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters, kernel_size, padding="same", dilation_rate=dilation, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    if skip.shape[-1] != filters:
        skip = layers.Conv1D(filters, 1, padding="same", use_bias=False)(skip)
    return layers.Add(name=f"tcn_b{idx}_add")([skip, x])

def build_tcn_small(input_shape: Sequence[int], num_classes: int, dropout: float = TCN_DROPOUT) -> keras.Model:
    """TCN-0.25: 16 filters, four residual blocks with dilations 1, 2, 4, 8, global average pooling, softmax.
    Names: input ``raw_input``, blocks ``tcn_b1_add``..``tcn_b4_add``, pooling ``tcn_gap``, head ``softmax``."""
    inputs = keras.Input(shape=_shape2(input_shape), name="raw_input")
    x = inputs
    for i, d in enumerate(TCN_DILATIONS, start=1):
        x = dilated_residual_block(x, TCN_FILTERS, kernel_size=TCN_KERNEL, dilation=d, dropout=dropout, idx=i)
    x = layers.GlobalAveragePooling1D(name="tcn_gap")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="softmax")(x)
    return keras.Model(inputs, outputs, name="TCN_Small")

# ----------------------------------------------------------------------------------------------------------------
# 2. TinierHAR
# ----------------------------------------------------------------------------------------------------------------
class AdditiveAttentionPooling(layers.Layer):
    """TinierHAR temporal head: ``a = softmax_t(Dense(1)(x))``, ``out = sum_t a_t x_t``; (B, T, D) -> (B, D)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.score = layers.Dense(1, name="score")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        a = tf.nn.softmax(self.score(x), axis=1)
        return tf.reduce_sum(a * x, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def _tinierhar_conv_block(x: tf.Tensor, cout: int, pool: bool, idx: int) -> tf.Tensor:
    """TinierHAR conv block: DepthwiseConv2D(5,1) -> Conv2D 1x1 -> BN -> ReLU [-> MaxPool(2,1)], plus a Conv2D 1x1 + BN
    shortcut on channel mismatch [-> MaxPool], added without activation (BN momentum / epsilon as in PyTorch)."""
    cin = int(x.shape[-1])
    m = layers.DepthwiseConv2D((TINIERHAR_KERNEL, 1), padding="same", depth_multiplier=1, use_bias=True,
                               name=f"th_b{idx}_dw")(x)
    m = layers.Conv2D(cout, 1, use_bias=True, name=f"th_b{idx}_pw")(m)
    m = layers.BatchNormalization(momentum=TORCH_BN_MOMENTUM, epsilon=TORCH_BN_EPSILON, name=f"th_b{idx}_bn")(m)
    m = layers.ReLU(name=f"th_b{idx}_relu")(m)
    if pool:
        m = layers.MaxPooling2D((2, 1), strides=(2, 1), padding="valid", name=f"th_b{idx}_pool")(m)
    s = x
    if cin != cout:
        s = layers.Conv2D(cout, 1, use_bias=True, name=f"th_b{idx}_sc_conv")(s)
        s = layers.BatchNormalization(momentum=TORCH_BN_MOMENTUM, epsilon=TORCH_BN_EPSILON,
                                      name=f"th_b{idx}_sc_bn")(s)
    if pool:
        s = layers.MaxPooling2D((2, 1), strides=(2, 1), padding="valid", name=f"th_b{idx}_sc_pool")(s)
    return layers.Add(name=f"th_b{idx}_add")([m, s])

def build_tinierhar(input_shape: Sequence[int], num_classes: int,
                    dropout: float = TINIERHAR_RELEASED_DROP_PROB) -> keras.Model:
    """TinierHAR (Bian et al., ISWC 2025), re-implemented from the paper: six conv blocks (two pooled 1->4->8, four
    8->8), feature-major flatten, Dropout, BiGRU(16), additive attention pooling ``tinierhar_feat``, Dense ``softmax``.
    Early-exit taps: ``th_b1_add``, ``th_b3_add``, ``th_b5_add``."""
    T, C = _shape2(input_shape)
    inputs = keras.Input(shape=(T, C), name="raw_input")
    x = layers.Reshape((T, C, 1), name="tinierhar_to_image")(inputs)
    x = _tinierhar_conv_block(x, TINIERHAR_NB_FILTERS, pool=True, idx=0)
    x = _tinierhar_conv_block(x, 2 * TINIERHAR_NB_FILTERS, pool=True, idx=1)
    for i in range(TINIERHAR_NB_CONV_BLOCKS):
        x = _tinierhar_conv_block(x, 2 * TINIERHAR_NB_FILTERS, pool=False, idx=2 + i)
    Tp, F = int(x.shape[1]), int(x.shape[3])
    if Tp < 1:
        raise ValueError(f"input T={T} too short for TinierHAR's two (2,1) poolings")
    x = layers.Permute((1, 3, 2), name="tinierhar_permute")(x)
    x = layers.Reshape((Tp, F * C), name="tinierhar_flatten")(x)
    if dropout and dropout > 0:
        x = layers.Dropout(dropout, name="tinierhar_drop")(x)
    h = layers.Bidirectional(layers.GRU(TINIERHAR_NB_UNITS_GRU, return_sequences=True, reset_after=True),
                             merge_mode="concat", name="tinierhar_bigru")(x)
    feat = AdditiveAttentionPooling(name="tinierhar_feat")(h)
    outputs = layers.Dense(num_classes, activation="softmax", name="softmax")(feat)
    return keras.Model(inputs, outputs, name="TinierHAR")

# ---------------------------------------------------------------------------------------------------------------
# 3. DeepConvLSTM a=0.25 (the ISWC22-HAR code's downsized DeepConvLSTM, F = int(0.25*64) = 16 filters, H = int(0.25*128) = 32 LSTM units, no BatchNorm)
# ---------------------------------------------------------------------------------------------------------------
DCL_NB_CONV_BLOCKS, DCL_NB_FILTERS, DCL_FILTER_WIDTH, DCL_DILATION = 2, 64, 5, 1 # configs/model.yaml `deepconvlstm:`

DCL_NB_LAYERS_LSTM, DCL_NB_UNITS_LSTM = 1, 128

def build_deepconvlstm_0_25(input_shape: Sequence[int], num_classes: int, dropout: float = 0.5,
                            alpha: float = 0.25, name: str = "DeepConvLSTM") -> keras.Model:
    """DeepConvLSTM at width 0.25: Reshape (T, C, 1) -> 2 x [Conv2D (5,1) stride 1 + ReLU, Conv2D (5,1) stride 2 + ReLU]
    -> Permute -> Reshape (T*, F*C) -> Dropout -> LSTM, last step ``dcl_feat`` -> Dense ``softmax``.  Early-exit taps:
    ``dcl_relu0_1``, ``dcl_relu1_0``, ``dcl_relu1_1``."""
    T, C = int(input_shape[0]), int(input_shape[1])
    a = float(alpha)
    F = int(a * DCL_NB_FILTERS)
    H = int(a * DCL_NB_UNITS_LSTM)
    inputs = keras.Input(shape=(T, C), name="raw_input")
    x = layers.Reshape((T, C, 1), name="dcl_to_image")(inputs)
    for b in range(DCL_NB_CONV_BLOCKS):
        for j, stride in enumerate((1, 2)):
            x = layers.Conv2D(F, (DCL_FILTER_WIDTH, 1), strides=(stride, 1), dilation_rate=(DCL_DILATION, 1),
                              padding="valid", use_bias=True, name=f"dcl_conv{b}_{j}")(x)
            x = layers.Activation("relu", name=f"dcl_relu{b}_{j}")(x)
    Ts = int(x.shape[1])
    if Ts < 1:
        raise ValueError(f"input T={T} too short for DeepConvLSTM's 4 valid convs")
    x = layers.Permute((1, 3, 2), name="dcl_permute")(x)
    x = layers.Reshape((Ts, F * C), name="dcl_flatten")(x)
    if dropout and dropout > 0:
        x = layers.Dropout(dropout, name="dcl_drop")(x)
    for i in range(DCL_NB_LAYERS_LSTM):
        last = (i == DCL_NB_LAYERS_LSTM - 1)
        x = layers.LSTM(H, return_sequences=not last, name="dcl_feat" if last else f"dcl_lstm{i}")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="softmax")(x)
    return keras.Model(inputs, outputs, name=name)


# early-exit taps: TCN residual blocks 1-3; TinierHAR conv blocks 2, 4 and 6
EXIT_TAPS: dict[str, list[str]] = {
    "tcn_small": ["tcn_b1_add", "tcn_b2_add", "tcn_b3_add"],
    "tinierhar": ["th_b1_add", "th_b3_add", "th_b5_add"],
    # DeepConvLSTM a=0.25: the outputs of convs 2, 3 and 4, i.e. every conv layer but the
    # first, before the single LSTM (27 % of the MACs on TORUS-9); the cascade never runs the LSTM before an early exit.
    "deepconvlstm_0_25": ["dcl_relu0_1", "dcl_relu1_0", "dcl_relu1_1"],
}


BUILDERS: dict[str, Callable[..., keras.Model]] = {"tinierhar": build_tinierhar, "tcn_small": build_tcn_small,
                                                   "deepconvlstm_0_25": build_deepconvlstm_0_25}


def build_backbone(fold, name: str) -> keras.Model:
    """keras.utils.set_random_seed(seed) immediately before the build, so the initial weights depend on the seed only."""
    from .config import DROPOUT
    keras.utils.set_random_seed(int(fold.seed))
    return BUILDERS[name](input_shape=(int(fold.X_tr.shape[1]), int(fold.X_tr.shape[2])), num_classes=int(fold.n_classes),
                          dropout=float(DROPOUT[name]))
