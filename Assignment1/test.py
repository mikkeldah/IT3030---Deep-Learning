import numpy as np

from nn import NeuralNetwork
from utils import sigmoid, add_bias_dimension


test_data_X = np.array([[1, 1], [2, 1]])
test_data_y = np.array([1, 2])



# NN with 5 input nodes, 20 hidden nodes and 8 output nodes
nn = NeuralNetwork(layers_array=np.array([2, 3, 1]))

hidden_layer = nn.layers[0]
output_layer = nn.layers[1]


# Testing for correct shapes of weights and biases
assert hidden_layer.weights.shape == (3, 3)

assert output_layer.weights.shape == (4, 1)


# Testing for correct output from forward pass

w1_test = np.array([
    [1.0, 0.5, 1.0],
    [0.5, 1.0, 0.5], 
    [1.0, 1.0, 1.0]
])

w2_test = np.array([
    [1.0],
    [0.5],
    [1.0], 
    [1.0]
])

# Test model
hidden_layer.set_weights(w1_test)
output_layer.set_weights(w2_test)
output = nn.forward_pass(test_data_X)

# Test outside model
test_data_X = add_bias_dimension(test_data_X)
output_1 = sigmoid(np.transpose(w1_test) @ test_data_X )

output_1 = add_bias_dimension(output_1)
output_true = sigmoid(np.transpose(w2_test) @ output_1)    


print(output)
print(output_true)
