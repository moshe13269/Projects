from abc import ABC

import tensorflow as tf
from tensorflow import keras
from typing import List, Tuple
import tensorflow_addons as tfa
from tensorflow.python.keras.layers import Layer as BaseLayer, Conv1D, Dropout

from dataclasses import dataclass, field


class ConvFeatureExtractionModel(BaseLayer, ABC):
    def __init__(self,
                 conv_layers: List[Tuple[int, int, int]],
                 dropout: float = 0.0,
                 mode: str = "default",
                 conv_bias: bool = False,
                 **kwargs):
        super().__init__(**kwargs)

        def block(n_out,
                  k,
                  stride,
                  is_layer_norm=False,
                  is_group_norm=False,
                  conv_bias=False):
            def make_conv():
                conv = Conv1D(filters=n_out, kernel_size=k, strides=stride, use_bias=conv_bias,
                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.))
                return conv

            assert (is_layer_norm and is_group_norm) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return keras.Sequential([
                    make_conv(),
                    Dropout(rate=dropout),
                    tf.keras.layers.LayerNormalization()
                    tf.keras.activations.gelu(),
                ])
            elif is_group_norm:
                return keras.Sequential([
                    make_conv(),
                    Dropout(rate=dropout),
                    tfa.layers.GroupNormalization(),
                    tf.keras.activations.gelu(),
                ])
            else:
                return keras.Sequential([make_conv(), Dropout(rate=dropout), tf.keras.activations.gelu()])

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def call(self, inputs, **kwargs):
        # BxT -> BxCxT
        inputs = inputs.unsqueeze(1)

        for conv in self.conv_layers:
            inputs = conv(inputs)

        return inputs
