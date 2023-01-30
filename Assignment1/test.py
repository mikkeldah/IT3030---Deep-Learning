import numpy as np

from neural_network import NeuralNetwork
from utils import sigmoid, softmax, cross_entropy_loss, add_bias_dimension

# Only for testing
from keras.datasets import mnist




test_data_X = np.array([[1, 1], [2, 1]])
test_data_y = np.array([1, 2])

test_data_X2 = np.array([
    [1, 2, 6, 4], 
    [2, 3, 1, 7], 
    [3, 5, 2, 3], 
    [1, 8, 9, 6]
])

test_data_y2 = np.array([0, 1, 1, 0])

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

def test_shapes(nn):
    hidden_layer = nn.layers[0]
    output_layer = nn.layers[1]
    
    # Testing for correct shapes of weights and biases
    assert hidden_layer.weights.shape == (3, 3)
    assert output_layer.weights.shape == (4, 1)


# Testing for correct output from forward pass
def test_forward_pass(nn, w1_test, w2_test, test_data_X):
    nn.layers[0].set_weights(w1_test)
    nn.layers[1].set_weights(w2_test)
    output = nn.forward_pass(test_data_X, test_data_y)

    # Test outside model
    test_data_X = add_bias_dimension(test_data_X)
    o1 = sigmoid(np.transpose(w1_test) @ test_data_X )

    o1 = add_bias_dimension(o1)
    output_true = sigmoid(np.transpose(w2_test) @ o1)    


    assert output.all() == output_true.all()
    assert np.isclose([output[0, 0], output[0, 1]], [0.9674060661, 0.9647823295], rtol=1e-6)[0]
    assert np.isclose([output[0, 0], output[0, 1]], [0.9674060661, 0.9647823295], rtol=1e-6)[1]

    print(output)


def network_test_without_softmax():
    nn = NeuralNetwork(layers_array=np.array([2, 3, 1]), softmax=False)
    test_shapes(nn)
    test_forward_pass(nn, w1_test, w2_test, test_data_X)

def network_test_with_softmax():
    nn2 = NeuralNetwork(layers_array=np.array([test_data_X2.shape[0], 6, 4, 3]), softmax=True)
    o = nn2.forward_pass(test_data_X2, test_data_y2)
    print(o)

if __name__ == "__main__":
    network_test_without_softmax()
    network_test_with_softmax()


   
