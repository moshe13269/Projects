from hydra.utils import instantiate


def acc_metrics_instantiate(to_metrics, outputs_dimension_per_outputs, metrics=None):
    if to_metrics:
        metrics_list = []
        for i in range(len(outputs_dimension_per_outputs)):
            metrics = instantiate(metrics)
            metrics.index_y_true = i
            metrics.num_classes = outputs_dimension_per_outputs[i]
            metrics.set_indexes()
            metrics_list.append(metrics)
        return metrics_list
    return None
