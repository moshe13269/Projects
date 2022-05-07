
from layers.encoder_blocks import Layer as EncoderBlock
from tensorflow.keras.layers import Layer as BaseLayer


class Layer(BaseLayer):

    def __init__(self, **kwargs):
        super(Layer, self).__init__(**kwargs)
        self.eb1 = EncoderBlock()
        self.eb2 = EncoderBlock()
        self.eb3 = EncoderBlock()
        self.eb4 = EncoderBlock()
        self.eb5 = EncoderBlock()
        self.eb6 = EncoderBlock()
        self.eb7 = EncoderBlock()

    def call(self, inputs, **kwargs):
        outputs = self.eb1(inputs)
        outputs = self.eb2(outputs)
        outputs = self.eb3(outputs)
        outputs = self.eb4(outputs)
        outputs = self.eb5(outputs)
        outputs = self.eb6(outputs)
        outputs = self.eb7(outputs)
        return outputs
