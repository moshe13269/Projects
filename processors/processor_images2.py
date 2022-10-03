import pickle
import tensorflow as tf


class Processor:
    t_axis: int
    prob2mask: float
    masking_size: int
    top_k_transformer: int

    def __init__(self,
                 t_axis: int,
                 prob2mask: float,
                 masking_size: int,
                 # top_k_transformer: int,
                 ):

        self.t_axis = t_axis
        self.prob2mask = prob2mask
        self.point2mask = 40 #int(self.prob2mask * self.t_axis)
        self.masking_size = masking_size

        self.patch_size = 32
        self.resnet = tf.keras.applications.resnet50.ResNet50(input_shape=(32, 32, 3), include_top=False)
        # self.top_k_transformer = top_k_transformer

    # the masking area is '1' and the unmasking by '0'
    def create_mask(self):
        rand_uniform = tf.random.uniform(maxval=1, shape=(self.t_axis,))
        mask = tf.where(
            tf.sign(rand_uniform - tf.reduce_min(tf.math.top_k(rand_uniform, k=self.point2mask)[0])) >= 0,
            1., 0.)
        return mask + tf.roll(mask, shift=1, axis=0)

    def load_data(self, path2data):
        if len(path2data) == 2:
            label_file = open(path2data[1], 'r')
            label = pickle.load(label_file)
            # todo: convert to onehot vector
        else:
            label = self.create_mask()

        image = tf.io.read_file(path2data[0])
        image = tf.image.decode_png(image, channels=3)

        patches = []
        # For square images only (as inputs.shape[1] = inputs.shape[2])
        input_image_size = image.shape[1]
        for i in range(0, input_image_size, self.patch_size):
            for j in range(0, input_image_size, self.patch_size):
                patches.append(tf.squeeze(self.resnet(
                    tf.expand_dims(
                        image[i: i + self.patch_size, j: j + self.patch_size, :], axis=0))))

        # image = tf.reshape(image, (image[0] * image[1], 3))
        # image = (wav_file - tf.reduce_mean(wav_file)) / tf.math.reduce_std(wav_file)
        # image = tf.audio.decode_wav(image)

        return [tf.convert_to_tensor(patches, dtype=tf.float32), label], [label]
