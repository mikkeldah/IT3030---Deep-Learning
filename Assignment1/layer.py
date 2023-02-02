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

        x = add_bias_dimension(x)
        self.activations = np.transpose(self.weights) @ x

        return self.activations

    def backward(self, J_acc):
        return




class ActivationLayer(Layer):
    def __init__(self, activation: Activation) -> None:
        self.activation_function = activation
    
    def forward(self, x):
        self.activations = self.activation_function(x)
        return self.activations

    def backward(self, J_acc):
        if self.activation_function == Activation.SIGMOID:
            J_sigmoid_z = self.activations * (1 - self.activations)
            J_sigmoid_z = np.diagflat(J_sigmoid_z.reshape(-1, 1))
            print("J_sigmoid_z: ", J_sigmoid_z)
            J_acc = J_acc * J_sigmoid_z
            return J_acc

        if self.activation_function == Activation.SOFTMAX:
            print("Activations: ", self.activations)
            sm = self.activations.reshape(-1, 1)
            print("Diagflat sm: ", np.diagflat(sm))
            print("np.dot(sm, sm.T): ", np.dot(sm, sm.T))
            J_softmax_z = np.diagflat(sm) - np.dot(sm, sm.T)
            print("J_soft: ", J_softmax_z)
            J_acc = J_acc * J_softmax_z
            print("J_acc:: ", J_acc)
            return J_acc 

        
        return
