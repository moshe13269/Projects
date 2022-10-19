
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model
from data2vec_image_model import Data2VecModel

# input_layer = Input(shape=(32 ,32 ,3,))
# x = LeNet5()(input_layer)

model = Model(inputs=Input(shape=(32, 32, 3,)), outputs=Data2VecModel()(Input(shape=(32, 32, 3,))))

print(model.summary(expand_nested=True))

