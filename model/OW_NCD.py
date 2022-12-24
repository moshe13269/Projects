
import tensorflow as tf
from typing import List, Tuple


class OWNCDModel:
    path2pretrain_model: str

    def __init__(self,
                 path2pretrain_model: str):
        self.pretrain_model = tf.keras.models.load_model(path2pretrain_model)

    def build(self):
        pass
