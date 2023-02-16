from Projects_tensorflow import layers
import tensorflow as tf
from typing import Tuple


class OWNCDModel:
    path2pretrain_model: str
    resnet: tf.keras.applications.resnet50

    def __init__(self,
                 path2pretrain_model: str,
                 inputs: Tuple[int, int, int, int],
                 resnet,
                 ):
        self.pretrain_model = tf.keras.models.load_model(path2pretrain_model)
        self.inputs = tf.keras.layers.Input(inputs)

        self.resnet = resnet
        self.split_negative_positive = layers.SplitNegativePositive()

    def build(self):
        outputs = self.resnet(self.inputs)
        prototype_u_candidates, prototype_l_candidates = self.split_negative_positive(outputs)
