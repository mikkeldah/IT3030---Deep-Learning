import numpy as np

from utils import sigmoid, add_bias_dimension, remove_bias_dimension

class NeuralNetwork:
    """
    The Network assumes that the first number of the input array is the size of the input layer
    and the last number of the input array is the size of the output layer. The rest of the 
    numbers are assumed to be the size of hidden layers in between. 

    The Network uses the same activation function for all of the hidden layers (sigmoid), and
    Softmax activation for the output layer. 

    """

    def __init__(self, layers_array: np.array):

        if not (1 < layers_array.size < 8):
            raise Exception("Network must contain at least one input and one output layer, and not more than 5 hidden layers. ")

        self.layers = []
        
        # Starting at the second layer, as there is no need to create an input layer object as it does not
        # have weights and biases associated with it
        for i in range(1, layers_array.size):
            self.layers.append(Layer(layer_size=layers_array[i], prev_layer_size=layers_array[i-1]))

    def forward_pass(self, x):

        x = add_bias_dimension(x)

        for layer in self.layers:
            x = add_bias_dimension(layer.forward(x))
        
        return remove_bias_dimension(x)

            
            
            

class Layer:
    def __init__(self, layer_size: int, prev_layer_size: int):
        # Adding a row to account for the bias trick
        self.weights = np.ones(shape=(prev_layer_size + 1, layer_size))

    def forward(self, x):
        self.outputs = sigmoid(np.transpose(self.weights) @ x)
        return self.outputs

    def set_weights(self, w):
        self.weights = w



