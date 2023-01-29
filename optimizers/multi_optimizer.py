from typing import List
import tensorflow as tf
from hydra.utils import instantiate
from tensorflow.keras.models import Model
from tensorflow_addons.optimizers import MultiOptimizer


class Optimizers:
    indexes: List[int]
    # optimizers_list: List[tf.keras.optimizers]
    model: Model
    index_transformer: int

    def __init__(self,
                 indexes,
                 optimizers_list,
                 ind_transformer_layer,
                 model=None):
        self.indexes = indexes
        self.optimizers_list = optimizers_list
        self.model = model
        self.ind_transformer_layer = ind_transformer_layer

    def crate_optimizers_list(self, model):

        optimizers_list = []

        for i in range(len(self.indexes)):
            if type(self.indexes[i]) != self.ind_transformer_layer:
                optimizers_list.append((self.optimizers_list[i], model.layers[self.indexes[i]]))
            else:
                for j in range(len(model.layers[self.indexes[i]].transformer_encoder.encoder_blocks)):

                    optimizers_list.append(
                        (self.optimizers_list[i],
                         model.layers[self.indexes[i]].transformer_encoder.encoder_blocks[j])
                    )

                    optimizers_list.append(
                        (self.optimizers_list[i],
                         model.layers[self.indexes[i]].transformer_decoder.decoder_blocks[j])
                    )
        return MultiOptimizer(optimizers_list)



"""
model.layers[1]:  layers.conv_wav_encoder.ConvFeatureExtractionModel
model.layers[4]:  layers.transformer_2.Transformer
model.layers[5]:  layers.conv_wav_decoder.ConvDecoderModel
model.layers[7]:  layers.linear_classifier_2.LinearClassifier
"""

"""
example:
https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/MultiOptimizer
https://www.tensorflow.org/api_docs/python/tfm/optimization/LinearWarmupConfig
https://towardsdatascience.com/advanced-techniques-for-fine-tuning-transformers-82e4e61e16e
"""