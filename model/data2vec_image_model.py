
import tensorflow as tf
from typing import List, Tuple
from layers.masking import Masking
from layers.ffn import FFN
from layers.transformer_encoder import TransformerEncoder
from layers.conv_image_encoder import ConvFeatureExtractionModel
from dataclasses import dataclass, field


class Data2VecModel(tf.keras.Model):
    # prob2mask: float
    # masking_length: int
    masking: bool
    masking_layer: Masking
    len_latent_space: int
    conv_encoder: ConvFeatureExtractionModel
    transformer_encoder: TransformerEncoder
    ffn: FFN
    tau: float
    top_k_transformer: int

    def __init__(self,
                 # prob2mask: float,
                 # masking_length: int,
                 masking: bool,
                 masking_layer: Masking,
                 len_latent_space: int,
                 conv_encoder: ConvFeatureExtractionModel,
                 transformer_encoder: TransformerEncoder,
                 ffn: FFN,
                 tau: float,
                 top_k_transformer: int
                 ):

        super(Data2VecModel, self).__init__()

        # self.prob2mask = prob2mask
        # self.masking_length = masking_length

        self.masking = masking  # bool
        self.masking_layer = masking_layer

        self.len_latent_space = len_latent_space
        self.conv_encoder = conv_encoder
        self.transformer_encoder = transformer_encoder
        self.ffn = ffn
        self.tau = tau
        self.top_k_transformer = top_k_transformer

    def __call__(self, inputs, **kwargs):

        image_file, mask = inputs

        teacher_inputs = tf.stop_gradient(tf.identity(image_file))

        if self.masking:
            masked_latent_space, mask = self.masking_layer([image_file, mask])
        else:
            masked_latent_space = image_file
            mask = tf.Variable([0])

        student_encoding = self.transformer_encoder(masked_latent_space,
                                                    training=True,
                                                    top_k_transformer=1)[0]

        teacher_encoding = tf.stop_gradient(self.transformer_encoder(teacher_inputs,
                                                                     training=False,
                                                                     top_k_transformer=self.top_k_transformer))

        teacher_encoding = tf.stop_gradient(tf.reduce_mean(self.ffn(teacher_encoding), axis=1))

        teacher_encoding = tf.stop_gradient(teacher_encoding * self.tau + (1 - self.tau) * student_encoding)

        return tf.concat([teacher_encoding, student_encoding], axis=1), mask


        # self.feature_extractor = ConvFeatureExtractionModel(conv_layers=cfg)
        # # conv_layers: List[Tuple[int, int, int]] = [(512, 10, 5), (512, 5, 3), (512, 3, 2)]
        # self.embed = cfg.encoder_embed_dim
        # self.average_top_k_layers = cfg.average_top_k_layers

    #     #######
    #     feature_enc_layers = eval(cfg.conv_feature_layers)
    #     self.extractor_embed = feature_enc_layers[-1][0]
    #
    #     self.ema = None
    #     self.embed = cfg.encoder_embed_dim
    #
    #     self.average_top_k_layers = cfg.average_top_k_layers
    #
    #     self.feature_extractor = ConvFeatureExtractionModel(
    #         conv_layers=feature_enc_layers,
    #         dropout=0.0,
    #         mode=cfg.extractor_mode,
    #         conv_bias=cfg.conv_bias,
    #     )
    #
    #     self.post_extract_proj = nn.Linear(self.extractor_embed, cfg.encoder_embed_dim)
    #
    #     self.mask_prob = cfg.mask_prob
    #     self.mask_selection = cfg.mask_selection
    #     self.mask_other = cfg.mask_other
    #     self.mask_length = cfg.mask_length
    #     self.no_mask_overlap = cfg.no_mask_overlap
    #     self.mask_min_space = cfg.mask_min_space
    #
    #     self.mask_channel_prob = cfg.mask_channel_prob
    #     self.mask_channel_before = cfg.mask_channel_before
    #     self.mask_channel_selection = cfg.mask_channel_selection
    #     self.mask_channel_other = cfg.mask_channel_other
    #     self.mask_channel_length = cfg.mask_channel_length
    #     self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
    #     self.mask_channel_min_space = cfg.mask_channel_min_space
    #
    #     self.dropout_input = nn.Dropout(cfg.dropout_input)
    #     self.dropout_features = nn.Dropout(cfg.dropout_features)
    #
    #     self.feature_grad_mult = cfg.feature_grad_mult
    #
    #     self.mask_emb = nn.Parameter(
    #         torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
    #     )
    #
    #     self.encoder = TransformerEncoder(cfg)
    #     self.layer_norm = LayerNorm(self.extractor_embed)
    #
    #     self.final_proj = nn.Linear(self.embed, self.embed)
    #
    #     self.num_updates = 0
    #
    # def make_ema_teacher(self):
    #     ema_config = EMAModuleConfig(
    #         ema_decay=self.cfg.ema_decay,
    #         ema_fp32=True,
    #     )
    #     skip_keys = set()
    #     if self.cfg.ema_layers_only:
    #         self.cfg.ema_transformer_only = True
    #         for k, _ in self.encoder.pos_conv.named_parameters():
    #             skip_keys.add(f"pos_conv.{k}")
    #
    #     self.ema = EMAModule(
    #         self.encoder if self.cfg.ema_transformer_only else self,
    #         ema_config,
    #         skip_keys=skip_keys,
    #     )
    #
    # def set_num_updates(self, num_updates):
    #     super().set_num_updates(num_updates)
    #
    #     if self.ema is None and self.final_proj is not None:
    #         logger.info(f"making ema teacher")
    #         self.make_ema_teacher()
    #     elif self.training and self.ema is not None:
    #         if self.cfg.ema_decay != self.cfg.ema_end_decay:
    #             if num_updates >= self.cfg.ema_anneal_end_step:
    #                 decay = self.cfg.ema_end_decay
    #             else:
    #                 decay = get_annealed_rate(
    #                     self.cfg.ema_decay,
    #                     self.cfg.ema_end_decay,
    #                     num_updates,
    #                     self.cfg.ema_anneal_end_step,
    #                 )
    #             self.ema.set_decay(decay)
    #         if self.ema.get_decay() < 1:
    #             self.ema.step(self.encoder if self.cfg.ema_transformer_only else self)
    #
    #     self.num_updates = num_updates
    #
    # def state_dict(self, destination=None, prefix="", keep_vars=False):
    #     state = super().state_dict(destination, prefix, keep_vars)
    #
    #     if self.ema is not None:
    #         state[prefix + "_ema"] = self.ema.fp32_params
    #
    #     return state
    #
    # def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
    #     if self.ema is not None:
    #         k = prefix + "_ema"
    #         assert k in state_dict
    #         self.ema.restore(state_dict[k], True)
    #         del state_dict[k]
    #     return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
    #
    # @classmethod
    # def build_model(cls, cfg: Data2VecAudioConfig, task=None):
    #     """Build a new data2vec_train_task instance."""
    #
    #     return cls(cfg)
    #
    # def apply_mask(self, x, padding_mask, mask_indices=None, mask_channel_indices=None):
    #     B, T, C = x.shape
    #
    #     if self.mask_channel_prob > 0 and self.mask_channel_before:
    #         mask_channel_indices = compute_mask_indices(
    #             (B, C),
    #             None,
    #             self.mask_channel_prob,
    #             self.mask_channel_length,
    #             self.mask_channel_selection,
    #             self.mask_channel_other,
    #             no_overlap=self.no_mask_channel_overlap,
    #             min_space=self.mask_channel_min_space,
    #         )
    #         mask_channel_indices = (
    #             torch.from_numpy(mask_channel_indices)
    #             .to(x.device)
    #             .unsqueeze(1)
    #             .expand(-1, T, -1)
    #         )
    #         x[mask_channel_indices] = 0
    #
    #     if self.mask_prob > 0:
    #         if mask_indices is None:
    #             mask_indices = compute_mask_indices(
    #                 (B, T),
    #                 padding_mask,
    #                 self.mask_prob,
    #                 self.mask_length,
    #                 self.mask_selection,
    #                 self.mask_other,
    #                 min_masks=1,
    #                 no_overlap=self.no_mask_overlap,
    #                 min_space=self.mask_min_space,
    #                 require_same_masks=self.cfg.require_same_masks,
    #                 mask_dropout=self.cfg.mask_dropout,
    #             )
    #             mask_indices = torch.from_numpy(mask_indices).to(x.device)
    #         x = index_put(x, mask_indices, self.mask_emb)
    #     else:
    #         mask_indices = None
    #
    #     if self.mask_channel_prob > 0 and not self.mask_channel_before:
    #         if mask_channel_indices is None:
    #             mask_channel_indices = compute_mask_indices(
    #                 (B, C),
    #                 None,
    #                 self.mask_channel_prob,
    #                 self.mask_channel_length,
    #                 self.mask_channel_selection,
    #                 self.mask_channel_other,
    #                 no_overlap=self.no_mask_channel_overlap,
    #                 min_space=self.mask_channel_min_space,
    #             )
    #             mask_channel_indices = (
    #                 torch.from_numpy(mask_channel_indices)
    #                 .to(x.device)
    #                 .unsqueeze(1)
    #                 .expand(-1, T, -1)
    #             )
    #         x = index_put(x, mask_channel_indices, 0)
    #
    #     return x, mask_indices
    #
    # def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
    #     """
    #     Computes the output length of the convolutional layers
    #     """
    #
    #     def _conv_out_length(input_length, kernel_size, stride):
    #         return torch.floor((input_length - kernel_size) / stride + 1)
    #
    #     conv_cfg_list = eval(self.cfg.conv_feature_layers)
    #
    #     for i in range(len(conv_cfg_list)):
    #         input_lengths = _conv_out_length(
    #             input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
    #         )
    #
    #     return input_lengths.to(torch.long)
    #
    # def forward(
    #     self,
    #     source,
    #     padding_mask=None,
    #     mask=True,
    #     features_only=False,
    #     layer=None,
    #     mask_indices=None,
    #     mask_channel_indices=None,
    #     padding_count=None,
    # ):
    #     features = source
    #
    #     if self.feature_grad_mult > 0:
    #         features = self.feature_extractor(features)
    #         if self.feature_grad_mult != 1.0:
    #             features = GradMultiply.apply(features, self.feature_grad_mult)
    #     else:
    #         with torch.no_grad():
    #             features = self.feature_extractor(features)
    #
    #     features = features.transpose(1, 2)
    #
    #     features = self.layer_norm(features)
    #
    #     orig_padding_mask = padding_mask
    #
    #     if padding_mask is not None and padding_mask.any():
    #         input_lengths = (1 - padding_mask.long()).sum(-1)
    #         # apply conv formula to get real output_lengths
    #         output_lengths = self._get_feat_extract_output_lengths(input_lengths)
    #
    #         padding_mask = torch.zeros(
    #             features.shape[:2], dtype=features.dtype, device=features.device
    #         )
    #
    #         # these two operations makes sure that all values
    #         # before the output lengths indices are attended to
    #         padding_mask[
    #             (
    #                 torch.arange(padding_mask.shape[0], device=padding_mask.device),
    #                 output_lengths - 1,
    #             )
    #         ] = 1
    #         padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
    #     else:
    #         padding_mask = None
    #
    #     if self.post_extract_proj is not None:
    #         features = self.post_extract_proj(features)
    #
    #     pre_encoder_features = None
    #     if self.cfg.ema_transformer_only:
    #         pre_encoder_features = features.clone()
    #
    #     features = self.dropout_input(features)
    #
    #     if mask:
    #         x, mask_indices = self.apply_mask(
    #             features,
    #             padding_mask,
    #             mask_indices=mask_indices,
    #             mask_channel_indices=mask_channel_indices,
    #         )
    #     else:
    #         x = features
    #         mask_indices = None
    #
    #     x, layer_results = self.encoder(
    #         x,
    #         padding_mask=padding_mask,
    #         layer=layer,
    #     )
    #
    #     if features_only:
    #         return {
    #             "x": x,
    #             "padding_mask": padding_mask,
    #             "layer_results": layer_results,
    #         }
    #
    #     result = {
    #         "losses": {},
    #     }
    #
    #     with torch.no_grad():
    #         self.ema.data2vec_train_task.eval()
    #
    #         if self.cfg.ema_transformer_only:
    #             y, layer_results = self.ema.data2vec_train_task.extract_features(
    #                 pre_encoder_features,
    #                 padding_mask=padding_mask,
    #                 min_layer=self.cfg.encoder_layers - self.average_top_k_layers,
    #             )
    #             y = {
    #                 "x": y,
    #                 "padding_mask": padding_mask,
    #                 "layer_results": layer_results,
    #             }
    #         else:
    #             y = self.ema.data2vec_train_task.extract_features(
    #                 source=source,
    #                 padding_mask=orig_padding_mask,
    #                 mask=False,
    #             )
    #
    #         target_layer_results = [l[2] for l in y["layer_results"]]
    #
    #         permuted = False
    #         if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
    #             target_layer_results = [
    #                 tl.permute(1, 2, 0) for tl in target_layer_results  # TBC -> BCT
    #             ]
    #             permuted = True
    #
    #         if self.cfg.batch_norm_target_layer:
    #             target_layer_results = [
    #                 F.batch_norm(
    #                     tl.float(), running_mean=None, running_var=None, training=True
    #                 )
    #                 for tl in target_layer_results
    #             ]
    #
    #         if self.cfg.instance_norm_target_layer:
    #             target_layer_results = [
    #                 F.instance_norm(tl.float()) for tl in target_layer_results
    #             ]
    #
    #         if permuted:
    #             target_layer_results = [
    #                 tl.transpose(1, 2) for tl in target_layer_results  # BCT -> BTC
    #             ]
    #
    #         if self.cfg.group_norm_target_layer:
    #             target_layer_results = [
    #                 F.layer_norm(tl.float(), tl.shape[-2:])
    #                 for tl in target_layer_results
    #             ]
    #
    #         if self.cfg.layer_norm_target_layer:
    #             target_layer_results = [
    #                 F.layer_norm(tl.float(), tl.shape[-1:])
    #                 for tl in target_layer_results
    #             ]
    #
    #         y = sum(target_layer_results) / len(target_layer_results)
    #
    #         if self.cfg.layer_norm_targets:
    #             y = F.layer_norm(y.float(), y.shape[-1:])
    #
    #         if self.cfg.instance_norm_targets:
    #             y = F.instance_norm(y.float().transpose(1, 2)).transpose(1, 2)
    #
    #         if not permuted:
    #             y = y.transpose(0, 1)
    #
    #         y = y[mask_indices]
    #
    #     x = x[mask_indices]
    #     x = self.final_proj(x)
    #
    #     sz = x.size(-1)
    #
    #     if self.loss_beta == 0:
    #         loss = F.mse_loss(x.float(), y.float(), reduction="none").sum(dim=-1)
    #     else:
    #         loss = F.smooth_l1_loss(
    #             x.float(), y.float(), reduction="none", beta=self.loss_beta
    #         ).sum(dim=-1)
    #
    #     if self.loss_scale is not None:
    #         scale = self.loss_scale
    #     else:
    #         scale = 1 / math.sqrt(sz)
    #
    #     result["losses"]["regression"] = loss.sum() * scale
    #
    #     if "sample_size" not in result:
    #         result["sample_size"] = loss.numel()
    #
    #     with torch.no_grad():
    #         result["target_var"] = self.compute_var(y)
    #         result["pred_var"] = self.compute_var(x.float())
    #
    #     if self.num_updates > 5000 and result["target_var"] < self.cfg.min_target_var:
    #         logger.error(
    #             f"target var is {result['target_var'].item()} < {self.cfg.min_target_var}, exiting"
    #         )
    #         raise Exception(
    #             f"target var is {result['target_var'].item()} < {self.cfg.min_target_var}, exiting"
    #         )
    #     if self.num_updates > 5000 and result["pred_var"] < self.cfg.min_pred_var:
    #         logger.error(
    #             f"pred var is {result['pred_var'].item()} < {self.cfg.min_pred_var}, exiting"
    #         )
    #         raise Exception(
    #             f"pred var is {result['pred_var'].item()} < {self.cfg.min_pred_var}, exiting"
    #         )
    #
    #     if self.ema is not None:
    #         result["ema_decay"] = self.ema.get_decay() * 1000
    #
    #     return result
    #
    # @staticmethod
    # def compute_var(y):
    #     y = y.view(-1, y.size(-1))
    #     if dist.is_initialized():
    #         zc = torch.tensor(y.size(0)).cuda()
    #         zs = y.sum(dim=0)
    #         zss = (y ** 2).sum(dim=0)
    #
    #         dist.all_reduce(zc)
    #         dist.all_reduce(zs)
    #         dist.all_reduce(zss)
    #
    #         var = zss / (zc - 1) - (zs ** 2) / (zc * (zc - 1))
    #         return torch.sqrt(var + 1e-6).mean()
    #     else:
    #         return torch.sqrt(y.var(dim=0) + 1e-6).mean()
    #
    # def extract_features(
    #     self, source, padding_mask, mask=False, layer=None
    # ):
    #     res = self.forward(
    #         source,
    #         padding_mask,
    #         mask=mask,
    #         features_only=True,
    #         layer=layer,
    #     )
    #     return res
    #
    # def remove_pretraining_modules(self, last_layer=None):
    #     self.final_proj = None
    #     self.ema = None
    #     if last_layer is not None:
    #         self.encoder.layers = nn.ModuleList(
    #             l for i, l in enumerate(self.encoder.layers) if i <= last_layer
    #         )
