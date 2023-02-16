import os
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import tensorflow_addons as tfa
from tensorflow.data import AUTOTUNE
# from tensorflow.python.keras.layers import Conv2D, ReLU, Input, Activation, Flatten, Dense

# tf.config_tensorflow.run_functions_eagerly(True)


def list_of_path2dataset(path2dataset):
    return [os.path.join(path2dataset, f) for f in listdir(path2dataset) if isfile(join(path2dataset, f))]


def label_generator():
    val = np.random.randint(10)
    label = np.concatenate([np.zeros(val), np.ones(1), np.zeros(9 - val)])
    return label


def map_function(path2sample):
    data = np.load(path2sample)
    label = label_generator()
    data = np.ndarray.astype(data, np.float32)
    label = np.ndarray.astype(label, np.float32)

    return data, label


class MyLayer(tf.keras.Model): #MyLayer(tf.keras.layers.Layer):

    def __init__(self, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.layer = self.create_layers()

    # @tf.custom_gradient
    def create_layers(self):

        def make_conv():
            return tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same')
        layers = []
        for i in range(self.num_layers):
            layer = tf.keras.Sequential([
                                make_conv(),
                                tf.keras.layers.Dropout(rate=0.3),
                                tf.keras.layers.LayerNormalization(),
                                tf.keras.layers.Activation(tf.nn.gelu), ])
            # if i==0:
            #     layer.build((None, 32, 32, 3))
            # else:
            #     layer.build((None, 32, 32, 512))
            layers.append(
                layer
            )

        # layers = tf.keras.Sequential(layers)
        # layers.build((None, 32, 32, 512))
        return layers

    def call(self, x):
        for layer in self.layer:
            x = layer(x)
        return x


class BaseModel:

    in1 = tf.keras.layers.Input(shape=(32, 32, 3,))
    in2 = tf.keras.layers.Input(shape=(10,))
    layers = MyLayer(3)
    conv1 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same')
    relu = tf.keras.layers.Activation(tf.nn.relu)
    bn = tf.keras.layers.LayerNormalization()
    conv2 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same')
    gn = tfa.layers.GroupNormalization()
    flatten = tf.keras.layers.Flatten()
    dense = tf.keras.layers.Dense(units=10)
    softmax = tf.keras.layers.Activation(tf.nn.softmax)

    @staticmethod
    def call():
        outputs = BaseModel.layers(BaseModel.in1)
        outputs = BaseModel.relu(BaseModel.conv1(outputs))
        outputs = BaseModel.gn(outputs)
        outputs = BaseModel.relu(BaseModel.conv2(outputs))
        outputs = BaseModel.bn(outputs)
        outputs = BaseModel.flatten(outputs)
        outputs = BaseModel.dense(outputs)
        outputs1 = tf.stop_gradient(BaseModel.in2)
        outputs = BaseModel.softmax(outputs)
        return tf.keras.Model(inputs=(BaseModel.in1, BaseModel.in2), outputs=(outputs, outputs1))


class SmoothL1Loss(tf.keras.losses.Loss):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()

    @tf.function
    def call(self, y_true, y_pred):
        x0 = y_pred[0]
        x1 = y_pred[1]
        res = tf.reduce_mean(tf.keras.metrics.mean_squared_error(x0, x1)*y_true)

        return res


class Train:
    def __init__(self, path2dataset):
        self.path2dataset = path2dataset
        self.dataset = tf.data.Dataset.from_tensor_slices((list_of_path2dataset(path2dataset)))
        self.dataset = (self.dataset.map(lambda inputs: tf.numpy_function(map_function, [inputs],
                                                                          [tf.float32, tf.float32])).map(lambda x, y: ((x, y), y)).batch(32).cache())
        self.basemodel = BaseModel
        self.model = self.basemodel.call()

    def run(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                           loss=SmoothL1Loss(), loss_weights=[1., 0.]) #SmoothL1Loss())
        a=0
        self.model.fit(x=self.dataset, epochs=7, verbose=1, batch_size=32)
        a=1


if __name__ == '__main__':
    # dataset_generator(num_samples=100, path2save=r'C:\Users\moshe\PycharmProjects\datasets\dataset_draft')
    dataset_path = list_of_path2dataset(r'C:\Users\moshe\PycharmProjects\datasets\dataset_draft')
    ds = tf.data.Dataset.from_tensor_slices(dataset_path)
    ds = ds.map(lambda item: tf.numpy_function(map_function, [item], [tf.float32, tf.float32]),
                num_parallel_calls=tf.data.AUTOTUNE).map(lambda x, y: ((x, y), y))
    train = Train(r'C:\Users\moshe\PycharmProjects\datasets\dataset_draft')
    train.run()



# def dataset_generator(num_samples, path2save):
#     for i in range(num_samples):
#         sample = np.random.normal(size=(32, 32, 3))
#         path = os.path.join(path2save, str(i))
#         np.save(arr=sample, file=path)
#     print('Dataset had been created')



# class BaseModel2(tf.keras.Model):
#
#     def __init__(self):
#         super().__init__()
#         self.inputs1 = tf.keras.layers.Input(shape=(32, 32, 3,))
#         self.inputs2 = tf.keras.layers.Input(shape=(16,))
#         self.conv1 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same')
#         self.conv2 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same')
#         self.relu = tf.keras.layers.Activation(tf.nn.relu)
#         self.flatten = tf.keras.layers.Flatten()
#         self.dense = tf.keras.layers.Dense(units=10)
#         self.softmax = tf.keras.layers.Activation(tf.nn.softmax)
#
#     def call(self, inputs, **kwargs):
#         inputs1 = self.input1(inputs[0])
#         inputs2 = self.inputs2(inputs[1])
#         outputs = self.conv1(inputs1)
#         outputs = self.relu(outputs)
#         tf.print(tf.shape(inputs))
#         outputs = self.conv2(outputs)
#         outputs = self.relu(outputs)
#         outputs = self.flatten(outputs)
#         outputs = self.dense(outputs)
#         outputs = self.softmax(outputs-inputs2)
#
#         return outputs


# import tensorflow as tf
# from tensorflow.python.keras.layers import Dense, Flatten, Conv2D
# from tensorflow.python.keras import Model

# class MyModel(Model):
#   def __init__(self):
#     super(MyModel, self).__init__()
#     self.layers_custom = self.create_layers()
#
#   def create_layers(self):
#     layers = [Conv2D(32, 3, activation='relu'), Flatten(),
#               Dense(128, activation='relu'), Dense(10, activation='softmax')]
#     return layers
#
#   def call(self, x, **kwargs):
#     for layer in self.layers_custom:
#       x = layer(x)
#     return x
#
# model = MyModel()