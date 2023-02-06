import numpy as np
import matplotlib.pyplot as plt

from utils import *

from enums import Activation


class Layer():
    def __init__():
        pass
        

class DenseLayer(Layer):
    def __init__(self, prev_layer_size: int, layer_size: int, activation: Activation) -> None:
        self.weights = np.random.uniform(low=-0.3, high=0.3, size=(prev_layer_size, layer_size))
        self.biases = np.ones((layer_size, 1))


    def forward(self, x: np.ndarray) -> np.ndarray:
        self.inputs = x

        self.sum = self.weights.T @ x + self.biases

        self.activations = sigmoid(self.sum)

        return self.activations

    def backward(self, J_L_z, lr, reglam, wrtype):

        # Sigmoid Jacobian
        J_z_sum = np.diagflat(self.activations * (1 - self.activations))

        # Jacobian from Z to Weights
        J_z_w = np.outer(self.inputs.T, J_z_sum.diagonal())

        # Compute dL/dw and dL/db
        J_L_w = J_L_z * J_z_w
        J_L_b = (J_L_z * J_z_sum.diagonal()).reshape(-1, 1)

        # Regularization
        if wrtype == 'l2':
            J_L_w = J_L_w + reglam * self.weights
            J_L_b = J_L_b + reglam * self.biases
        elif wrtype == 'l1':
            J_L_w = J_L_w + reglam * np.sign(self.weights)
            J_L_b = J_L_b + reglam * np.sign(self.biases)

        # Jacobian from this to previous layer
        J_z_y = J_z_sum @ self.weights.T

        # Compute dL/dY
        J_L_y = J_L_z @ J_z_y

        # Updating weights and biases
        self.weights = self.weights - lr * J_L_w
        self.biases = self.biases - lr * J_L_b

        return J_L_y
    


class SoftmaxLayer(Layer):
    def __init__(self) -> None:
        pass
    
    def forward(self, x):
        self.activations = softmax(x)
        return self.activations

    
    def backward(self, J_acc, lr, reglam, wrtype):

        s = self.activations
        J_softmax_z = np.diagflat(s) - np.outer(s, s)

        J_acc = J_acc @ J_softmax_z

        return J_acc
