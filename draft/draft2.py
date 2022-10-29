import os
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
from tensorflow.data import AUTOTUNE
# from tensorflow.python.keras.layers import Conv2D, ReLU, Input, Activation, Flatten, Dense

tf.config.run_functions_eagerly(True)


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


class BaseModel:

    @staticmethod
    def call():
        inputs1 = tf.keras.layers.Input(shape=(32, 32, 3,))
        inputs2 = tf.keras.layers.Input(shape=(10,))
        outputs = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same')(inputs1)
        outputs = tf.keras.layers.Activation(tf.nn.relu)(outputs)
        outputs = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same')(outputs)
        outputs = tf.keras.layers.Activation(tf.nn.relu)(outputs)
        outputs = tf.keras.layers.Flatten()(outputs)
        outputs = tf.keras.layers.Dense(units=10)(outputs)
        outputs = tf.keras.layers.Activation(tf.nn.softmax)(outputs-inputs2)

        return tf.keras.Model(inputs=[inputs1, inputs2], outputs=outputs)


class BaseModel2(tf.keras.Model):

    def __init__(self):
        super().__init__()
        # self.inputs1 = tf.keras.layers.Input(shape=(32, 32, 3,))
        # self.inputs2 = tf.keras.layers.Input(shape=(10,))
        # self.inputs = tf.c
        self.conv1 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same')
        self.relu = tf.keras.layers.Activation(tf.nn.relu)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10)
        self.softmax = tf.keras.layers.Activation(tf.nn.softmax)

    def call(self, inputs, **kwargs):
        # inputs1 = self.input1(inputs)
        outputs = self.conv1(inputs)
        outputs = self.relu(outputs)
        tf.print(tf.shape(inputs))
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        outputs = self.flatten(outputs)
        outputs = self.dense(outputs)
        outputs = self.softmax(outputs)

        return outputs


class Train:
    def __init__(self, path2dataset):
        self.path2dataset = path2dataset
        self.dataset = tf.data.Dataset.from_tensor_slices((list_of_path2dataset(path2dataset)))
        self.dataset = (self.dataset.map(lambda inputs: tf.numpy_function(map_function, [inputs],
                                                                          [tf.float32, tf.float32])).map(lambda x, y: ((x, y), y)).batch(32).cache())
        self.basemodel = BaseModel
        self.model = self.basemodel.call()
        # self.model2 = BaseModel2()

    def run(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                           loss=tf.keras.losses.categorical_crossentropy)
        self.model.fit(x=self.dataset, epochs=4, verbose=1, batch_size=10)


if __name__ == '__main__':
    # dataset_generator(num_samples=100, path2save=r'C:\Users\moshe\PycharmProjects\datasets\dataset_draft')
    dataset_path = list_of_path2dataset(r'C:\Users\moshe\PycharmProjects\datasets\dataset_draft')
    ds = tf.data.Dataset.from_tensor_slices(dataset_path)
    ds = ds.map(lambda item: tf.numpy_function(map_function, [item], [tf.float32, tf.float32]),
                num_parallel_calls=tf.data.AUTOTUNE).map(lambda x, y: ((x, y), y))

    output_shapes = (([2, 2], [2, 3]), [2, 5])
    # ds = ds.flat_map()
        # .map(lambda x, y: tf.data.Dataset.from_tensor_slices(tf.stack([tf.broadcast_to(x, (repeats,)), y], axis=1)))
    # print(ds.__iter__().__next__())
    # a = ds.__iter__().__next__()

    train = Train(r'C:\Users\moshe\PycharmProjects\datasets\dataset_draft')
    train.run()

# def dataset_generator(num_samples, path2save):
#     for i in range(num_samples):
#         sample = np.random.normal(size=(32, 32, 3))
#         path = os.path.join(path2save, str(i))
#         np.save(arr=sample, file=path)
#     print('Dataset had been created')
