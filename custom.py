
class Minloss_checkpoint:

    def __init__(self, period):

        self.min_loss = 10 ** 32
        self.min_vloss = 10 ** 32

        self.period = period

    def __call__(self, epoch, _, vloss):

        save = False
        
        if self.period and epoch % self.period == 0:
            save = save or True

        if vloss < self.min_vloss:
            self.min_vloss = vloss
            save = save or True

        return save