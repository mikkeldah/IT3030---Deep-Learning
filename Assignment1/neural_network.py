import numpy as np

from utils import sigmoid, softmax, add_bias_dimension, remove_bias_dimension

class NeuralNetwork:
    """
    The Network assumes that the first number of the input array is the size of the input layer
    and the last number of the input array is the size of the output layer. The rest of the 
    numbers are assumed to be the size of hidden layers in between. 

    The Network uses the same activation function for all of the hidden layers (sigmoid), and
    Softmax activation for the output layer. 

    """

    def __init__(self, layers_array: np.array, softmax: bool):
        if not (1 < layers_array.size < 8):
            raise Exception("Network must contain at least one input and one output layer, and not more than 5 hidden layers. ")

        self.softmax = softmax
        self.layers = []
        
        # Starting at the second layer, as there is no need to create an input layer object as it does not
        # have weights and biases associated with it
        for i in range(1, layers_array.size):
            self.layers.append(Layer(layer_size=layers_array[i], prev_layer_size=layers_array[i-1]))

    def forward_pass(self, x, y):

        # Add row of ones to the input matrix to account for the bias trick
        x = add_bias_dimension(x)

        for layer in self.layers:
            x = layer.forward(x)
            x = add_bias_dimension(x)
        
        # Remove bias row from the final output
        x = remove_bias_dimension(x)

        if self.softmax:
            return softmax(x)

        return x

    def backward_pass(self):

        for layer in reversed(self.layers):
            layer.backward()
            
            
            

class Layer:
    def __init__(self, layer_size: int, prev_layer_size: int):
        # Random init of weights in range [0, 1]. Adding a row to account for the bias trick.
        self.weights = np.random.rand(prev_layer_size + 1, layer_size)

    def forward(self, x):
        """
        Args:
            x: inputs of shape [previous layer size, batch size]
        Returns:
            self.outputs: output of layer with shape [layer size, batch size]

        """
        self.inputs = x
        self.outputs = sigmoid(np.transpose(self.weights) @ x)

        return self.outputs 
    
    def backward(self):
        
        

        return


    def set_weights(self, w):
        self.weights = w



