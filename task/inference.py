import os
import dataset
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

        self.outputs_dimension_per_outputs = cfg.train_task.TrainTask.outputs_dimension_per_outputs
        self.index2split = [sum(self.outputs_dimension_per_outputs[:i])
                            for i in range(len(self.outputs_dimension_per_outputs) + 1)]

        self.path2save_csv = self.cfg.train_task.TrainTask.get('path2save_csv')

        self.path2load_model = self.cfg.train_task.TrainTask.get('path2load_model')
        self.model = tf.keras.models.load_model(self.path2load_model, compile=False)

        self.processor = instantiate(cfg.train_task.TrainTask.processor)

        self.batch_size = self.cfg.train_task.TrainTask.get('batch_size')

        self.results = instantiate(cfg.train_task.TrainTask.results)

        self.train_dataset, self.test_dataset, self.val_dataset = dataset.split_dataset(self.dataset_class)

        train_dataset = dataset.data_loader_inference(self.train_dataset,
                                                      self.processor.load_data,
                                                      self.processor.mask_inference,
                                                      1,
                                                      1)

        test_dataset = dataset.data_loader_inference(self.test_dataset,
                                                     self.processor.load_data,
                                                     self.processor.mask_inference,
                                                     1,
                                                     1)

        val_dataset = dataset.data_loader_inference(self.val_dataset,
                                                    self.processor.load_data,
                                                    self.processor.mask_inference,
                                                    1,
                                                    1)

        self.dataset_ = {'train': train_dataset, 'test': test_dataset, 'val': val_dataset}

        self.num_outputs = self.cfg.train_task.TrainTask.get('num_outputs')

    def evaluate_model(self, dataset_, dim_vector_output, name):

        num_sample = int(dataset_.__len__().numpy())
        dataset_iter = dataset_.as_numpy_iterator()

        results = np.zeros((num_sample * 2, dim_vector_output))
        accuracy = np.zeros(dim_vector_output)

        for sample in range(num_sample):
            x, y = dataset_iter.next()
            y_ = self.model.predict(x)
            y_true, y_pred = self.convert_outputs2classes(y_true=y, y_pred=y_)

            results[2 * sample: 2 * sample + 1, :] = y_true
            results[2 * sample + 1: 2 * sample + 2, :] = y_pred

            for j in range(y_true.shape[0]):
                if y_true[j] == y_pred[j]:
                    accuracy[j] += 1

            if sample == 1000:
                break;

        pd.DataFrame(results).to_csv(os.path.join(self.path2save_csv, name + 'csv_results.csv'))

        accuracy = accuracy / 1000.
        print(accuracy)
        print(accuracy.mean())
        np.save(arr=accuracy, file=self.path2save_csv + 'accuracy')

    def split(self, inputs):
        return [inputs[:, self.index2split[i]: self.index2split[i + 1]] for i in range(len(self.index2split) - 1)]

    def convert_outputs2classes(self, y_true, y_pred):
        y_true = self.split(y_true[0])
        y_pred = self.split(y_pred[1])

        y_true_list = []
        y_pred_list = []
        for i in range(len(y_pred)):
            y_true_list.append(tf.argmax(y_true[i][0]).numpy())
            y_pred_list.append(tf.argmax(tf.nn.softmax(y_pred[i])[0]).numpy())
        return np.asarray(y_true_list), np.asarray(y_pred_list)

    def run(self):
        with tf.device('/GPU:3'):
            for dataset_name in self.dataset_.keys():
                self.evaluate_model(self.dataset_[dataset_name], self.num_outputs, dataset_name)
