import os
import numpy as np
import pandas as pd
import tensorflow as tf
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split


class Inference:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self.dataset_class = instantiate(cfg.train_task.TrainTask.dataset_class)

        self.test_dataset = None

        self.path2save_csv = self.cfg.train_task.TrainTask.get('path2save_csv')

        self.model_name = self.cfg.train_task.TrainTask.get('model_name')
        self.model = instantiate(cfg.train_task.TrainTask.model)

        self.processor = instantiate(cfg.train_task.TrainTask.processor)

        self.batch_size = self.cfg.train_task.TrainTask.get('batch_size')

        self.results = instantiate(cfg.train_task.TrainTask.results)

    @staticmethod
    def convert_outputs_model_2_parmas(outputs):
        params_list = []
        for param in outputs:
            param = param.squeeze()
            param = tf.nn.softmax(param)
            params_list.append(tf.argmax(param).numpy())
        return np.asarray(params_list)

    def evaluate_model(self, model, test_set):

        num_sample = int(test_set.__len__().numpy())
        test_set = test_set.as_numpy_iterator()

        results = np.zeros((num_sample * 2, 16))

        for sample in range(num_sample):
            x, y = test_set.next()
            y_ = model.predict_on_batch(x)  # model.predict(x)
            y_ = Inference.convert_outputs_model_2_parmas(y_)
            results[2 * sample: 2 * sample + 1, :] = y_.squeeze()
            results[2 * sample + 1: 2 * sample + 2, :] = y[0].squeeze()

        pd.DataFrame(results).to_csv(os.path.join(self.path2save_csv, 'csv_results.csv'))

    def run(self):

        dataset_had_split = len(self.dataset_class.dataset_names_train) > 0 and \
                            len(self.dataset_class.dataset_names_test) > 0

        if dataset_had_split:
            X_test = self.dataset_class.dataset_names_test
            y_test = self.dataset_class.labels_names_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.dataset_class.dataset_names,
                                                                self.dataset_class.labels_names,
                                                                test_size=0.2,
                                                                random_state=1)

        self.test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        test_dataset = (self.test_dataset
                        .shuffle(1024)
                        .map(
            lambda path2data, path2label: tf.numpy_function(self.processor.load_data, [(path2data, path2label)],
                                                            [tf.float32, tf.float32])
            , num_parallel_calls=tf.data.AUTOTUNE)
                        .cache()
                        .batch(self.batch_size['test'])
                        .prefetch(tf.data.AUTOTUNE)
                        )

        with tf.device('/GPU:3'):
            self.evaluate_model(self.model, test_dataset)
