
class Minloss_checkpoint:

    def __init__(self, when):

        self.min_loss = 10 ** 32
        self.min_vloss = 10 ** 32

        self.when = when

    def __call__(self, epoch, _, vloss):

        save = False
        
        if epoch in self.when:
            save = save or True

        if vloss < self.min_vloss:
            self.min_vloss = vloss
            save = save or True

        return save

class Lr_absolute_sched:

    def __init__(self, lr_dict):

        self.lr_dict = lr_dict

    def __call__(self, epoch, _):

        return self.lr_dict[epoch]

# TODO lr that increases / decreases relative to current lr

class Lr_gradual_sched1:

    def __init__(self, rate, epochs):

        self.rate = rate
        self.epochs = epochs

    def __call__(self, epoch, lr):
        
        if epoch < self.epochs:
            return lr * self.rate

        return lr
