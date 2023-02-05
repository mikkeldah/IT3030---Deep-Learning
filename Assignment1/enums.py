from enum import Enum

from utils import *

class Activation(Enum):
    SIGMOID = sigmoid
    SOFTMAX = softmax
    NONE = ""

class Cost(Enum):
    CROSS_ENTROPY_LOSS = cross_entropy_loss