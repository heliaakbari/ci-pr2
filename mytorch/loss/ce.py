from mytorch import Tensor
from ..activation import softmax
import numpy as np

def CategoricalCrossEntropy(preds: Tensor, label: Tensor):
    "TODO: implement Categorical Cross Entropy loss"
    _preds = softmax(preds)
    _sum = (label * _preds).sum()
    size = Tensor(np.ndarray(preds.shape).fill(label.shape[0]))
    size = size**-1
    return _sum * size
