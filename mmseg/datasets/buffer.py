import ctypes
import multiprocessing as mp
from typing import Dict

import numpy as np


class SharedValue(object):

    def __init__(self, init_value: float):
        self.init_value = init_value
        self.val = mp.Value(ctypes.c_float, init_value)

    def update(self, iters: int):
        with self.val.get_lock():
            alpha = min(1 - 1 / (iters + 1), self.init_value)
            self.val.value = alpha

    @property
    def value(self):
        return self.val.value


class FixedBuffer:
    """Abstraction that holds a numpy array and uses it as circular buffer.
    """

    def __init__(self, num_classes: int, alpha: float = 0.968):
        self.shared_base = mp.Array(ctypes.c_float, num_classes)
        self.shared_array = np.ctypeslib.as_array(self.shared_base.get_obj())
        self.alpha = SharedValue(init_value=alpha)
        self.num_classes = num_classes
        self.index = 0

    def append(self, data: Dict[int, float], iters: int):
        with self.shared_base.get_lock():  # synchronize access
            self.alpha.update(iters)
            alpha = self.alpha.value
            arr = np.ctypeslib.as_array(self.shared_base.get_obj())  # no data copying
            for c, v in data.items():
                arr[c] = arr[c] * alpha + (1 - alpha) * v

    def get_counts(self):
        with self.shared_base.get_lock():  # synchronize access
            arr = np.ctypeslib.as_array(self.shared_base.get_obj())  # no data copying
        return arr
