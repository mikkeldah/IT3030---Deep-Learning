import numpy as np

from utils import *
from layer import Layer

from enums import * 


class NeuralNetwork:

    def __init__(self, cost_function: Cost):
        self.layers = []
        self.cost_function = cost_function

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

        loss = cross_entropy_loss(targets, outputs)

        J_cost_output = -np.sum(targets / outputs, axis=1)
        print("J_cost_output: ", J_cost_output * np.array([[1, 1], [1, 1]]))

        J_acc = J_cost_output

        for layer in reversed(self.layers):
            J_acc = layer.backward(J_acc) 
            
            
        


