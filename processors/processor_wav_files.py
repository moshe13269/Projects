
import pickle
import numpy as np
import tensorflow as tf
from scipy.io import wavfile


class Processor:
    t_axis: int
    prob2mask: float
    masking_size: int
    top_k_transformer: int
    load_label: bool

    def __init__(self,
                 t_axis: int,
                 prob2mask: float,
                 masking_size: int,
                 load_label: bool,
                 ):

        self.t_axis = t_axis
        self.prob2mask = prob2mask
        self.point2mask = int(self.prob2mask * self.t_axis)
        self.masking_size = masking_size
        self.load_label = load_label

    # the masking area is '1' and the unmasking by '0'
    def create_mask(self):
        rand_uniform = np.random.uniform(high=1., size=(self.t_axis,))
        #top k:
        indexes_top_k = np.argpartition(rand_uniform, -self.point2mask)[-self.point2mask:]
        top_k = rand_uniform[indexes_top_k]
        min_from_top_k = np.nanmin(top_k, axis=-1)
        mask = np.where(np.sign(rand_uniform - min_from_top_k) >= 0, 1., 0.)
        return np.ndarray.astype(mask, np.float32)

    def load_data(self, path2data):
        if self.load_label:
            path2label = path2data.replace('data', 'labels')
            label_file = open(path2label, 'r')
            label = pickle.load(label_file)
        #     # todo: convert to onehot vector
        else:
            label = self.create_mask()

        samplerate, data = wavfile.read(path2data)

        data = np.ndarray.astype(data, np.float32)
        label = np.ndarray.astype(label, np.float32)

        return data, label


if __name__ == '__main__':
    from dataset.dataset import Dataset

    dataset = Dataset(path2load=r'C:\Users\moshe\PycharmProjects\datasets\cifar10',
                      labels=False, type_files='npy', dataset_names='cifar10')
    processor = Processor(t_axis=16, prob2mask=0.1, masking_size=1, load_label=False)
    train_dataset = tf.data.Dataset.from_tensor_slices(dataset.dataset_names_train[:10])
    train_dataset = (train_dataset
                     .shuffle(1024)
                     .map(processor.load_data, num_parallel_calls=tf.data.AUTOTUNE)
                     .cache()
                     .repeat()
                     .batch(2)
                     .prefetch(tf.data.AUTOTUNE)
                     )
    iterator = train_dataset.as_numpy_iterator()
    for i in range(1):
        d = iterator.next()[0][0]
        print(d)
        print(d.shape)