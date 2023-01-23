from typing import List
import tensorflow as tf
from hydra.utils import instantiate
from tensorflow.keras.models import Model
from tensorflow_addons.optimizers import MultiOptimizer


class Optimizers:
    learning_rates: List[float]
    optimizers: List[tf.keras.optimizers]
    model: Model

    @staticmethod
    def crate_optimizers_list(#learning_rates,
                              optimizers,
                              model):

        optimizers_list = []

        for i in range(len(optimizers)):
            optimizers_list.append((instantiate(optimizers[i]), model.layers[i]))
        return MultiOptimizer(optimizers_list)


"""
example:
https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/MultiOptimizer
"""