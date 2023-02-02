import numpy as np

from utils import *
from neural_network import NeuralNetwork
from enums import Activation, Cost
from layer import *


X = np.array([[1, 2], [3, 4]])
y = np.array([[0, 1], [1, 0]])


nn = NeuralNetwork(cost_function=Cost.CROSS_ENTROPY_LOSS)

nn.add_layer(DenseLayer(prev_layer_size=2, layer_size=2))
nn.add_layer(ActivationLayer(Activation.SIGMOID))
nn.add_layer(DenseLayer(prev_layer_size=2, layer_size=2))
nn.add_layer(ActivationLayer(Activation.SIGMOID))
nn.add_layer(ActivationLayer(Activation.SOFTMAX))


# Forward pass - output.shape is [batch size, num_classes]
output = nn.forward_pass(X)

print("Output Forward Pass: ", output)

# Loss
loss = cross_entropy_loss(y, output)
print("Cross Entropy Loss: ", loss)

# Backward pass
nn.backward_pass(output, y)

