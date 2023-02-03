import numpy as np

from utils import *

from enums import Activation


class Layer():
    def __init__():
        pass
        

class DenseLayer(Layer):
    def __init__(self, prev_layer_size: int, layer_size: int) -> None:
        # Random init of weights in range [0, 1]. Adding a row to account for the bias trick.
        #self.weights = np.random.rand(prev_layer_size + 1, layer_size)
        self.weights = np.ones((prev_layer_size + 1, layer_size))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Expects an activation matrix of size [input size, batch size]

        """
        print("W shape: ", self.weights.shape)
        print("x shape: ", x.shape)

        # Add a 1 to each training example to account for the bias trick
        x = add_bias_dimension(x)

        # Need for the backward pass
        self.input = x

        self.activations = np.transpose(self.weights) @ x

        return self.activations

    def backward(self, J_acc, learning_rate):

        print("J_acc input to Dense Layer: ", J_acc)
        print("Shape: ", J_acc.shape)
        print("Weights (dz/dw): ", self.weights)
        print("Weights (dz/dw) shape: ", self.weights.shape)

        # Add dz/dx to the accumulated Jacobian
        J_acc = J_acc @ self.weights ############ FIX

        
        batch_size = self.input.shape[1]
        # dC/dw Jacobian
        J_cost_weights = (1 / batch_size) * J_acc.diagonal().reshape(-1, self.input.shape[1]) @ self.input.T
        print("J_cost_weights: ", J_cost_weights)

        
        print("Weights before: ", self.weights)
        # Update Weights 
        self.weights = self.weights - learning_rate * J_cost_weights.T

        print("Weights after: ", self.weights)

        return J_acc




class ActivationLayer(Layer):
    def __init__(self, activation: Activation) -> None:
        self.activation_function = activation
    
    def forward(self, x):
        self.activations = self.activation_function(x)
        return self.activations

    def backward(self, J_acc, learning_rate):
        if self.activation_function == Activation.SIGMOID:
            J_sigmoid_z = self.activations * (1 - self.activations)
            J_sigmoid_z = np.diagflat(J_sigmoid_z.reshape(-1, 1))
            print("J_sigmoid (ds/dz): ", J_sigmoid_z)
            J_acc = J_acc * J_sigmoid_z
            print("J_acc after Sigmoid: ", J_acc)
            return J_acc

        if self.activation_function == Activation.SOFTMAX:
            sm = self.activations.reshape(-1, 1)
            J_softmax_z = np.diagflat(sm) - np.dot(sm, sm.T)
            J_acc = J_acc * J_softmax_z
            print("J_acc after Softmax: ", J_acc)
            return J_acc 

        
        return
