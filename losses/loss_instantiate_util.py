from hydra.utils import instantiate


def ce_loss_instantiate(outputs_dimension_per_outputs, loss_ce):
    loss_list = []
    for i in range(len(outputs_dimension_per_outputs)):
        loss = instantiate(loss_ce)
        loss.index_y_true = i
        loss.num_classes = outputs_dimension_per_outputs[i]
        loss.set_indexes()
        loss_list.append(loss)
    return loss_list


def losses_instantiate(num_ce_loss, loss_ce, outputs_dimension_per_outputs, loss=None):
    if num_ce_loss == 1:
        loss_ce = instantiate(loss_ce)
    else:
        loss_ce = ce_loss_instantiate(list(outputs_dimension_per_outputs))

    if loss is not None:
        audio_loss = list(instantiate(loss))
        return audio_loss + list(loss_ce)
    return list(loss_ce)
