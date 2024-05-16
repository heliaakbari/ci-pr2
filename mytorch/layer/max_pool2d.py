from mytorch import Tensor
from mytorch.layer import Layer

import numpy as np

class MaxPool2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)) -> None:
        super()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        Nh, Nw = x.shape # input size
        Kh, Kw = self.kernel_size # Kernel size (along height and width)
        sh, sw = self.stride

        Oh = (Nh-Kh)//sh + 1 # output height
        Ow = (Nw-Kw)//sw + 1 # output width

        # creating appropriate strides
        strides = (sh*Nw, sw, Nw, 1) 
        strides = tuple(i * x.itemsize for i in strides) 

        subM = np.lib.stride_tricks.as_strided(x, shape=(Oh, Ow, Kh, Kw),
                                       strides=strides)
        
        return np.max(subM, axis=(2,3))
    
    def __str__(self) -> str:
        return "max pool 2d - kernel: {}, stride: {}, padding: {}".format(self.kernel_size, self.stride, self.padding)
