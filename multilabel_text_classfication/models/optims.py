import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from transformers.optimization import AdamW

class Optim(object):

    def set_parameters(self, params):
        self.params = list(params)  # careful: params may be a generator
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr)
        elif self.method=='bertadam':
            self.optimizer = AdamW(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, method, lr, max_grad_norm, lr_decay=1, start_decay_at=None, max_decay_times=2):
        self.last_score = None
        self.decay_times = 0
        self.max_decay_times = max_decay_times
        self.lr = float(lr)
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False

    def step(self):
        # Compute gradients norm.
        if self.max_grad_norm:
            #梯度裁剪
            clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    #如果val perf没有改善，或者我们达到start_decay_at极限，则衰减学习率
    def updateLearningRate(self, score, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)

        self.last_score = score
        self.optimizer.param_groups[0]['lr'] = self.lr
