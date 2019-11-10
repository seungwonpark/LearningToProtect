import os
from tensorboardX import SummaryWriter


class MyWriter():
    def __init__(self, hp, logdir):
        self.logdir = logdir
        self.ab = self.create_writer('AB')
        self.bob = self.create_writer('Bob')
        self.eve = self.create_writer('Eve')
        self.rand = self.create_writer('random')

    def create_writer(self, name):
        writer = SummaryWriter(os.path.join(self.logdir, name))
        return writer

    def log_train(self, loss_ab, loss_b, loss_e, step):
        self.ab.add_scalar('loss', loss_ab, step)
        self.bob.add_scalar('loss', loss_b, step)
        self.eve.add_scalar('loss', loss_e, step)

    def log_accuracy(self, acc_b, acc_e, step):
        self.bob.add_scalar('accuracy', acc_b, step)
        self.eve.add_scalar('accuracy', acc_e, step)
        self.rand.add_scalar('accuracy', 0.5, step)
