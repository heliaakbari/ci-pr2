from typing import List
from mytorch.layer import Layer
from mytorch.optimizer import Optimizer
from .. import Tensor

class SGD(Optimizer):
    def __init__(self, layers:List[Layer], learning_rate=0.1):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
        for l in self.layers:
            l.weight = l.weight - l.weight.grad * Tensor([self.learning_rate])
            if l.need_bias:
                l.bias = l.bias - l.bias.grad * Tensor([self.learning_rate])