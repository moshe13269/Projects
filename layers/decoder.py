
from layers.decoder_blocks import Layer as DecoderBlock
from tensorflow.keras.layers import Layer as BaseLayer


class Layer(BaseLayer):

    def __init__(self, **kwargs):
        super(Layer, self).__init__(**kwargs)
        self.db1 = DecoderBlock()
        self.db2 = DecoderBlock()
        self.db3 = DecoderBlock()
        self.db4 = DecoderBlock()
        self.db5 = DecoderBlock()
        self.db6 = DecoderBlock()
        self.db7 = DecoderBlock()

    def call(self, inputs, **kwargs):
        outputs = self.db1(inputs)
        outputs = self.db2(outputs)
        outputs = self.db3(outputs)
        outputs = self.db4(outputs)
        outputs = self.db5(outputs)
        outputs = self.db6(outputs)
        outputs = self.db7(outputs)
        return outputs
