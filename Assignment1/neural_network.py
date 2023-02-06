import numpy as np
import matplotlib.pyplot as plt

from utils import *
from layer import Layer

from enums import * 


class NeuralNetwork:

    def __init__(self, cost_function: Cost, lr: float, reglam: float = 0, wrtype: str = None):
        self.layers = []
        self.cost_function = cost_function
        self.lr = lr
        self.reglam = reglam
        self.wrtype = wrtype

        print("Initialized Network with the following parameters: ")
        print("Cost function: ", cost_function)
        print("Learning rate: ", lr)
        print("Regularization factor: ", reglam)
        print("Regularization type: ", wrtype)


    def add_layer(self, layer: Layer) -> None: 
        self.layers.append(layer)

    def set_layers(self, layers):
        self.layers = layers
        print("Number of hidden layers: ", len(self.layers) - 2)


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

        J_acc = cost_grad

        for layer in reversed(self.layers):
            J_acc = layer.backward(J_acc, self.lr, self.reglam, self.wrtype) 
        
            
        
    def train(self, features, targets):

        losses = []
        losses_show = []

        # For each training example repeat this process:
        for i in range(features.shape[0]):

            # Preprosess data
            x = features[i].flatten().reshape(-1, 1)
            y = targets[i].reshape(1,-1)

            # Forward pass - output.shape is [batch size, num_classes]
            output = self.forward_pass(x)

            # Backward pass
            self.backward_pass(output, y)

            # Validation and Loss
            loss = cross_entropy_loss(y, output)
            losses.append(loss)

            if len(losses) > 20:
                losses_show.append(np.mean(losses[-20:]))
            else:
                losses_show.append(losses[-1])


        plt.plot(range(len(losses_show)), losses_show)
        plt.show()

    def predict(self, x):
        x = x.flatten().reshape(-1, 1)

        pred = self.forward_pass(x)

        max_index = np.argmax(pred)
        pred = np.zeros(pred.size)
        pred[max_index] = 1
        return pred
