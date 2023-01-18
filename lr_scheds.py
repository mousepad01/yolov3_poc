import numpy as np
import tensorflow as tf

class Lr_constant:

    def __init__(self):
        pass

    def __call__(self, epoch, batch_idx, lr):
        return lr

class Lr_dict_sched:

    def __init__(self, lr_dict):

        self.lr_dict = lr_dict

    def __call__(self, epoch, batch_idx, lr):
        return self.lr_dict[epoch]

class Lr_linear_decay:

    def __init__(self, rate, epochs):

        self.rate = rate
        self.epochs = epochs

    def __call__(self, epoch, batch_idx, lr):
        
        if epoch < self.epochs:
            return lr * self.rate

        return lr

class Lr_cosine_decay:

    def __init__(self, min_lr, max_lr, period, b2e):

        self.period = period

        self.min_lr = min_lr
        self.max_lr = max_lr

        self.PI = tf.constant(np.pi)

        self.b2e = b2e

    def __call__(self, epoch, batch_idx, lr):

        epoch %= self.period

        epoch += batch_idx * self.b2e
        return self.min_lr + 1 / 2 * (self.max_lr - self.min_lr) * (1 + tf.cos(self.PI * (epoch / self.period)))
    
class Lr_cosine_decay_warmup:

    def __init__(self, min_lr, max_lr, period, b2e, warmup_bcnt):

        self.period = period

        self.min_lr = min_lr
        self.max_lr = max_lr

        self.PI = tf.constant(np.pi)

        self.b2e = b2e

        self.warmup_ecnt = warmup_bcnt * self.b2e
        assert(self.warmup_ecnt < self.period)
        self.period -= self.warmup_ecnt

    def __call__(self, epoch, batch_idx, lr):

        epoch %= self.period
        epoch += batch_idx * self.b2e

        if epoch <= self.warmup_ecnt:
            return (epoch / self.warmup_ecnt) * self.max_lr
        
        epoch -= self.warmup_ecnt
        return self.min_lr + 1 / 2 * (self.max_lr - self.min_lr) * (1 + tf.cos(self.PI * (epoch / self.period)))

class Triangular_rate_policy:

    def __init__(self, min_lr, max_lr, stepsize, b2e):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.stepsize = stepsize

        self.b2e = b2e
        
    def __call__(self, epoch, batch_idx, lr):

        epoch += batch_idx * self.b2e

        cycle = tf.floor(1 + epoch / (2 * self.stepsize))
        x = tf.abs(epoch / self.stepsize - 2 * cycle + 1)

        return self.min_lr + (self.max_lr - self.min_lr) * max(0, 1 - x)
