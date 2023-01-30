import tensorflow as tf


class WarmLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule): #tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self,
                 initial_learning_rate,
                 warmup_steps,
                 hold_step,
                 decay_step,
                 max_learn_rate,
                 min_learn_rate
                 ):
        super(WarmLRSchedule).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_step = hold_step
        self.decay_step = decay_step
        self.max_learn_rate = max_learn_rate
        self.min_learn_rate = min_learn_rate

    def __call__(self, step):

        step = tf.cast(step, dtype=tf.float32)

        if self.warmup_steps-step >= 0.:
            lr = self.initial_learning_rate + \
                step * ((self.max_learn_rate - self.initial_learning_rate) / self.warmup_steps)
            # tf.keras.backend.set_value(self.model.optimizer.lr, lr)
            return lr
        elif (self.hold_step + self.warmup_steps) - step >= 0.:
            # tf.keras.backend.set_value(self.model.optimizer.lr, self.max_learn_rate)
            return self.max_learn_rate
        num_step = self.hold_step + self.warmup_steps - step
        lr = self.max_learn_rate - num_step * ((self.min_learn_rate - self.max_learn_rate) / self.decay_step)
        # tf.keras.backend.set_value(self.model.optimizer.lr, max(self.min_learn_rate, lr))
        return max(self.min_learn_rate, lr)

"""
τ linearly increases from τ₀ to a target value τₑ 
for the first τₙ updates and then stays constant 
for the remaining of the training
"""


class EMACallback(tf.keras.callbacks.Callback):
    t_n: int
    t_0: float
    t_e: float

    def __init__(self,
                 t_n: int,
                 t_0: float,
                 t_e: float,
                 ):
        super(EMACallback, self).__init__()
        self.tau = t_0
        self.t_n = t_n
        self.t_0 = t_0
        self.t_e = t_e
        self.step = (self.t_e - self.t_0) / self.t_n
        # self.tau = self.t_0 - self.step

    def on_train_batch_begin(self, batch, logs=None):

        if self.tau < self.t_e:
            self.tau += self.step

        # self.model.transformer_encoder_s.get_weights()
        #
        # layer_b.set_weights(layer_a.get_weights())

        # student transformer: model.layers[5];  teacher transformer: model.layers[6]
        teacher_weight = self.model.layers[6].get_weights()
        student_weight = self.model.layers[5].get_weights()
        for i in range(len(teacher_weight)):
            teacher_weight[i] = teacher_weight[i] * self.tau + student_weight[i] * (1. - self.tau)

        self.model.layers[6].set_weights(teacher_weight)
        self.model.layers[6].pos_embedding.set_weights(self.model.layers[5].pos_embedding.get_weights())

        # tf.keras.backend.set_value(self.model.layers[6].get_weights(),
        #                            self.model.layers[5].get_weights() * self.tau +
        #                            self.model.layers[6].get_weights() * (1.-self.tau))

        # tf.keras.backend.set_value(self.model.layers[6].pos_embedding.get_weights(),
        #                            self.model.layers[5].pos_embedding.get_weights())


# class LearnRateSchedulerTriStage(tf.keras.callbacks.Callback):
#     initial_lr: float
#     peak: float
#     end: float
#     steps2warp_up: float
#     steps2hold: float
#     decay: float
#     train_steps_per_epoch: int
#     epochs: int
#
#     def __init__(self,
#                  initial_lr: float,
#                  peak: float,
#                  end: float,
#                  steps2warp_up: float,  # precent
#                  steps2hold: float,  # precent
#                  decay: float,  # precent
#                  train_steps_per_epoch: int,
#                  epochs: int,
#                  ):
#         super(LearnRateSchedulerTriStage, self).__init__()
#         assert steps2warp_up + steps2hold + decay == 1., "steps2warp_up + steps2hold + decay = %f" % \
#                                                          (steps2warp_up + steps2hold + decay)
#         self.initial_lr = initial_lr
#         self.peak = peak
#         self.num_steps2warp_up = int(steps2warp_up * (epochs * train_steps_per_epoch))
#         self.num_steps2hold = int(steps2hold * (epochs * train_steps_per_epoch))
#         self.num_decay = int(decay * (epochs * train_steps_per_epoch))
#         self.steps_counter = 0
#         self.stage = 0
#         self.end = end
#
#     def on_train_batch_begin(self, batch, logs=None):
#         self.steps_counter += 1
#
#         if self.steps_counter < self.num_steps2warp_up:
#             self.stage = 0
#             # tf.print('stage: %d' % self.stage)
#         elif self.steps_counter >= self.num_steps2warp_up and self.steps_counter <= self.num_decay + self.num_steps2warp_up:
#             self.stage = 1
#             # tf.print('stage: %d' % self.stage)
#         else:
#             self.stage = 2
#             # tf.print('stage: %d' % self.stage)
#
#         lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
#
#         if self.stage == 0:
#             lr += (self.peak - self.initial_lr) / self.num_steps2warp_up
#         elif self.stage == 2:
#             lr -= (self.peak - self.initial_lr) / self.num_decay
#
#         tf.keras.backend.set_value(self.model.optimizer.lr, lr)

        # keys = list(logs.keys())
        # # print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

# class LearnRateSchedulerTriStage(tf.keras.callbacks.Callback):
#     initial_lr: float
#     peak: float
#     steps2warp_up: float
#     steps2hold: float
#     decay: float
#     train_steps_per_epoch: int
#     epochs: int
#
#     def __init__(self,
#                  initial_lr: float,
#                  peak: float,
#                  steps2warp_up: float,  # precent
#                  steps2hold: float,  # precent
#                  decay: float,  # precent
#                  train_steps_per_epoch: int,
#                  epochs: int,
#                  ):
#         super(LearnRateSchedulerTriStage, self).__init__()
#         assert steps2warp_up + steps2hold + decay == 1., "steps2warp_up + steps2hold + decay = %f" % \
#                                                          (steps2warp_up + steps2hold + decay)
#         self.initial_lr = initial_lr
#         self.peak = peak
#         self.num_steps2warp_up = int(steps2warp_up * (epochs * train_steps_per_epoch))
#         self.num_steps2hold = int(steps2hold * (epochs * train_steps_per_epoch))
#         self.num_decay = int(decay * (epochs * train_steps_per_epoch))
#         self.steps_counter = 0
#         self.stage = 0
#
#     def on_train_batch_begin(self, batch, logs=None):
#         self.steps_counter += 1
#
#         if self.steps_counter < self.num_steps2warp_up:
#             self.stage = 0
#         elif self.steps_counter >= self.num_steps2warp_up and self.steps_counter <= self.num_decay + self.num_steps2warp_up:
#             self.stage = 1
#         else:
#             self.stage = 2
#
#         lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
#
#         if self.stage == 0:
#             lr += (self.peak - self.initial_lr) / self.num_steps2warp_up
#         elif self.stage == 2:
#             lr -= (self.peak - self.initial_lr) / self.num_decay
#
#         tf.keras.backend.set_value(self.model.optimizer.lr, lr)
