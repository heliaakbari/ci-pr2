from mytorch import Tensor
from mytorch.layer import Layer
from mytorch.util import initializer

import numpy as np

class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), 
                 need_bias: bool = False, mode="xavier") -> None:
        super().__init__(need_bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.need_bias = need_bias
        self.weight: Tensor = None
        self.bias: Tensor = None
        self.initialize_mode = mode

        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        self.x = x
        batch_size, in_channel, input_width, input_height = x.shape
        input_width_new = int(1 + (input_width - self.kernel_size) / self.stride)
        input_height_new = int(1 + (input_height - self.kernel_size) / self.stride)
        out = initializer((batch_size, self.out_channels, input_width_new, input_height_new),"zero")
        for i in range(input_width_new):
            for j in range(input_height_new):
                x_windows = x[:, :, i * self.stride: i * self.stride + self.kernel_size,
                            j * self.stride: j * self.stride + self.kernel_size]
                for k in range(batch_size):
                    for l in range(self.out_channels):
                        out[k, l, i, j] = np.sum(x_windows[k] * self.weight[l])
                        out[k, l, i, j] += self.bias[l]
        return out
    
    def initialize(self):
        self.weight = Tensor(
            data=initializer((self.in_channels, self.out_channels), self.initialize_mode),
            requires_grad=True
        )
        if self.need_bias:
            self.bias = Tensor(
                data=initializer((1, self.outputs),"zero"),
                requires_grad=True
            )

    def zero_grad(self):
        "TODO: implement zero grad"
        self.weight.zero_grad()
        if self.need_bias:
            self.bias.zero_grad()

    def parameters(self):
        "TODO: return weights and bias"
        if self.need_bias:
            return [self.weight, self.bias]
        return [self.weight]
    
    def __str__(self) -> str:
        return "conv 2d - total params: {} - kernel: {}, stride: {}, padding: {}".format(
                                                                                    self.kernel_size[0] * self.kernel_size[1],
                                                                                    self.kernel_size,
                                                                                    self.stride, self.padding)
