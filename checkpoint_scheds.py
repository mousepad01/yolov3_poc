
class No_checkpoint:

    def __init__(self):
        pass

    def __call__(self):
        return False

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
