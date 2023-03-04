import tensorflow as tf
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def torch_data_loader(dataset_list, processor_function, batch_size):
    return DataLoader(dataset_list,
                      batch_size=batch_size,
                      collate_fn=processor_function,
                      pin_memory=True)


def data_loader(dataset_list,
                processor_function1,
                processor_function2,
                num_outputs,
                batch_size):
    dataset = (
        dataset_list
        .shuffle(dataset_list.cardinality().numpy(), reshuffle_each_iteration=True)
        .map(
            lambda path2data, path2label: tf.numpy_function(processor_function1, [(path2data, path2label)],
                                                            [tf.float32, tf.float32])
            , num_parallel_calls=tf.data.AUTOTUNE).map(
            lambda x, y: (processor_function2(x), tuple([y for i in range(num_outputs)])))
        .cache()
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .repeat()
    )

    return dataset


def data_loader_inference(dataset_list,
                          processor_function1,
                          processor_function2,
                          num_outputs,
                          batch_size):
    dataset = (
        dataset_list
        .shuffle(dataset_list.cardinality().numpy(), reshuffle_each_iteration=True)
        .map(
            lambda path2data, path2label: tf.numpy_function(processor_function1, [(path2data, path2label)],
                                                            [tf.float32, tf.float32])
            , num_parallel_calls=tf.data.AUTOTUNE).map(
            lambda x, y: (processor_function2(x), tuple([y for i in range(num_outputs)])))
        .cache()
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
        # .repeat()
    )

    return dataset


def split_dataset(dataset_class):
    dataset_had_split = len(dataset_class.dataset_names_train) > 0 and \
                        len(dataset_class.dataset_names_test) > 0

    if dataset_had_split:
        x_train = dataset_class.dataset_names_train
        y_train = dataset_class.labels_names_train
        x_test = dataset_class.dataset_names_test
        y_test = dataset_class.labels_names_test
    else:
        x_train, x_test, y_train, y_test = train_test_split(dataset_class.dataset_names,
                                                            dataset_class.labels_names,
                                                            test_size=0.2,
                                                            random_state=1)

    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      test_size=0.1,
                                                      random_state=1)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    return train_dataset, test_dataset, val_dataset
