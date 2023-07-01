from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def ce_loss_instantiate(outputs_dimension_per_outputs, loss_ce):
    loss_list = []
    for i in range(len(outputs_dimension_per_outputs)):
        loss = instantiate(loss_ce)
        loss.index_y_true = i
        loss.num_classes = outputs_dimension_per_outputs[i]
        loss.set_indexes()
        loss_list.append(loss)
    return loss_list


def losses_instantiate(num_ce_loss=None, loss_ce=None, outputs_dimension_per_outputs=None, loss=None):
    if num_ce_loss is not None:
        if num_ce_loss == 1:
            loss_ce = [instantiate(loss_ce)]
        else:
            loss_ce = ce_loss_instantiate(list(outputs_dimension_per_outputs))

    # if len(loss_ce) == 1:
    #     loss_ce = [loss_ce]

    if loss is not None:
        # num_losses = len(loss)
        audio_loss = instantiate(loss)
        # if num_losses == 1:
        #     audio_loss = [audio_loss]
        if num_ce_loss is None:
            return audio_loss
        return audio_loss + loss_ce
    return list(loss_ce)