import tensorflow as tf

"""
τ linearly increases from τ₀ to a target value τₑ 
for the first τₙ updates and then stays constant 
for the remaining of the training
"""


class EMACallback(tf.keras.callbacks.Callback):
    t_n: float  # int
    t_0: float
    t_e: float

    def __init__(self,
                 t_n: float,
                 t_0: float,
                 t_e: float,
                 ):
        super(EMACallback, self).__init__()
        self.t_n = t_n
        self.t_0 = t_0
        self.t_e = t_e
        self.step = (self.t_e - self.t_0) / self.t_n
        self.tau = self.t_0 - self.step

    # def on_batch_begin(self, logs=None, **kwargs):
    #     self.tau = min(self.t_e, self.t_0 + self.step)


class LearnRateScheduler(tf.keras.callbacks.Callback):
    t_n: float  # int
    t_0: float
    t_e: float

    def __init__(self,
                 t_n: float,
                 t_0: float,
                 t_e: float,
                 ):
        super(LearnRateScheduler, self).__init__()
        self.t_n = t_n
        self.t_0 = t_0
        self.t_e = t_e
        self.step = (self.t_e - self.t_0) / self.t_n
        self.tau = self.t_0 - self.step

    # def on_batch_begin(self, logs=None, **kwargs):
    #     self.tau = min(self.t_e, self.t_0 + self.step)