import pickle
import numpy as np
import tensorflow as tf


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
                 # top_k_transformer: int,
                 ):

        self.t_axis = t_axis
        self.prob2mask = prob2mask
        self.point2mask = int(self.prob2mask * self.t_axis)
        self.masking_size = masking_size
        self.load_label = load_label
        # self.top_k_transformer = top_k_transformer

    # the masking area is '1' and the unmasking by '0'
    @tf.function(autograph=True)
    def create_mask(self):
        rand_uniform = tf.random.uniform(maxval=1, shape=(self.t_axis,))
        mask = tf.where(
            tf.sign(rand_uniform - tf.reduce_min(tf.math.top_k(rand_uniform, k=self.point2mask)[0])) >= 0.,
            1., 0.)
        return mask

    # @tf.function(input_signature=[tf.TensorSpec(None, tf.string)]) # @tf.function(autograph=True)
    def load_data(self, path2data):
        if self.load_label:
            path2label = path2data.replace('data', 'labels')
            label_file = open(path2label, 'r')
            label = pickle.load(label_file)
            # todo: convert to onehot vector
        else:
            label = self.create_mask()

        image = np.load(path2data)

        image = np.ndarray.astype(image, np.float32)
        label = np.ndarray.astype(label, np.float32)
        return (image, label), (label)


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

        # path2data = tf.strings.split(path2data, sep='$')
        # print(path2data)

        # image = np.load(path2data, allow_pickle=True)
        # image.astype(np.float32)
        # image = tf.convert_to_tensor(image)

        # dtype = tf.float32
        #
        # header_offset = Processor.npy_header_offset(path2data)
        # image = tf.data.FixedLengthRecordDataset([path2data], 3 * dtype.size, header_bytes=header_offset)

        # image = tf.io.read_file(path2data)
        # image = tf.io.decode_raw(image, tf.uint8)

        # image = tf.Tensor(image, dtype=tf.float32, value_index=3)
        # image = tf.convert_to_tensor(image)
        # print(image)
        import os
        # print(tf.strings.split(path2data, os.path.sep))
        # image = tf.image.decode_png(image, channels=3)
        # image.set_shape([32, 32, 3])
        # if tf.reduce_max(image) > 1.:
        #     image = image / 255.

        # image = tf.py_func(np.load(path2data))
        # image = tf.function(np.load, path2data, tf.string)
        # image = tf.numpy_function(np.load, [path2data], tf.float32) #np.load(path2data)

   # @staticmethod
   #  def npy_header_offset(npy_path):
   #      with open(str(npy_path), 'rb') as f:
   #          if f.read(6) != b'\x93NUMPY':
   #              raise ValueError('Invalid NPY file.')
   #          version_major, version_minor = f.read(2)
   #          if version_major == 1:
   #              header_len_size = 2
   #          elif version_major == 2:
   #              header_len_size = 4
   #          else:
   #              raise ValueError('Unknown NPY file version {}.{}.'.format(version_major, version_minor))
   #          header_len = sum(b << (8 * i) for i, b in enumerate(f.read(header_len_size)))
   #          header = f.read(header_len)
   #          if not header.endswith(b'\n'):
   #              raise ValueError('Invalid NPY file.')
   #          return f.tell()