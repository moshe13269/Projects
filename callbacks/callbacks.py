import tensorflow as tf

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
        self.t_n = t_n
        self.t_0 = t_0
        self.t_e = t_e
        self.step = (self.t_e - self.t_0) / self.t_n
        # self.tau = self.t_0 - self.step

    def on_train_batch_begin(self, batch, logs=None):

        tau = float(tf.keras.backend.get_value(self.model.tau))
        if tau < self.t_e:
            tau += self.step

        tf.keras.backend.set_value(self.model.tau, tau)


class LearnRateSchedulerTriStage(tf.keras.callbacks.Callback):
    initial_lr: float
    peak: float
    steps2warp_up: float
    steps2hold: float
    decay: float
    train_steps_per_epoch: int
    epochs: int

    def __init__(self,
                 initial_lr: float,
                 peak: float,
                 steps2warp_up: float,  # precent
                 steps2hold: float,  # precent
                 decay: float,  # precent
                 train_steps_per_epoch: int,
                 epochs: int,
                 ):
        super(LearnRateSchedulerTriStage, self).__init__()
        assert steps2warp_up + steps2hold + decay == 1., "steps2warp_up + steps2hold + decay = %f" % \
                                                         (steps2warp_up + steps2hold + decay)
        self.initial_lr = initial_lr
        self.peak = peak
        self.num_steps2warp_up = int(steps2warp_up * (epochs * train_steps_per_epoch))
        self.num_steps2hold = int(steps2hold * (epochs * train_steps_per_epoch))
        self.num_decay = int(decay * (epochs * train_steps_per_epoch))
        self.steps_counter = 0
        self.stage = 0

    def on_train_batch_begin(self, batch, logs=None):
        self.steps_counter += 1

        if self.steps_counter < self.num_steps2warp_up:
            self.stage = 0
        elif self.steps_counter >= self.num_steps2warp_up and self.steps_counter <= self.num_decay + self.num_steps2warp_up:
            self.stage = 1
        else:
            self.stage = 2

        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

        if self.stage == 0:
            lr += (self.peak - self.initial_lr) / self.num_steps2warp_up
        elif self.stage == 2:
            lr -= (self.peak - self.initial_lr) / self.num_decay

        tf.keras.backend.set_value(self.model.optimizer.lr, lr)

        # keys = list(logs.keys())
        # # print("...Training: start of batch {}; got log keys: {}".format(batch, keys))


