from enum import Enum

from utils import sigmoid, softmax, relu, cross_entropy_loss

class Activation(Enum):
    SIGMOID = sigmoid
    SOFTMAX = softmax
    RELU = relu
    NONE = ""

class Cost(Enum):
    CROSS_ENTROPY_LOSS = cross_entropy_loss

