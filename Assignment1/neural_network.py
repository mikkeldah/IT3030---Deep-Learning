import numpy as np

from utils import *
from layer import Layer

from enums import * 


class NeuralNetwork:

    def __init__(self, cost_function: Cost, learning_rate: float):
        self.layers = []
        self.cost_function = cost_function
        self.learning_rate = learning_rate

    def add_layer(self, layer: Layer) -> None: 
        self.layers.append(layer)

    def forward_pass(self, x: np.ndarray) -> np.ndarray:   
        """
        params
            x: initial input to the network with shape [input size, batch size]

        return: the final output of the network with shape [batch size, num classes]

        """

        for layer in self.layers:
            x = layer.forward(x)

        return np.transpose(x)

    def backward_pass(self, outputs, targets):

        cost_grad = cross_entropy_loss_grad(targets, outputs)
        print("J_cost_output: ", cost_grad)

        J_acc = cost_grad

        for layer in reversed(self.layers):
            J_acc = layer.backward(J_acc, self.learning_rate) 
            
            
        


