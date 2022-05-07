
from tensorflow.keras.layers import Layer as BaseLayer, Conv2DTranspose, BatchNormalization
from tensorflow.keras.activations import gelu


class Layer(BaseLayer):

    def __init__(self, **kwargs):
        super(Layer, self).__init__(**kwargs)
        self.conv1 = Conv2DTranspose()
        self.gn1 = BatchNormalization(axis=1)
        self.conv2 = Conv2DTranspose()
        self.gn2 = BatchNormalization(axis=1)

    def call(self, inputs, **kwargs):
        outputs = gelu(self.gn1(self.conv1(inputs)))
        outputs = gelu(self.gn2(self.conv2(outputs)))
        return outputs
