import numpy as np
import tensorflow as tf

class Lr_constant:

    def __init__(self):
        pass

    def __call__(self, _, lr):
        return lr

class Lr_dict_sched:

    def __init__(self, lr_dict):

        self.lr_dict = lr_dict

    def __call__(self, epoch, _):
        return self.lr_dict[epoch]

class Lr_linear_decay:

    def __init__(self, rate, epochs):

        self.rate = rate
        self.epochs = epochs

    def __call__(self, epoch, lr):
        
        if epoch < self.epochs:
            return lr * self.rate

        return lr

class Lr_cosine_decay:

    def __init__(self, min_lr, max_lr, period):

        self.period = period

        self.min_lr = min_lr
        self.max_lr = max_lr

        self.PI = tf.constant(np.pi)

    def __call__(self, epoch, _):
        return self.min_lr + 1 / 2 * (self.max_lr - self.min_lr) * (1 + tf.cos(self.PI * (epoch / self.period)))
