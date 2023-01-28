import numpy as np

def sigmoid(z: float):
    return 1 / (1 + np.exp(-z))

def add_bias_dimension(x: np.array):
    biases = np.ones((1, x.shape[1]))
    return np.append(x, biases, axis=0)

def remove_bias_dimension(x: np.array):
    return x[:-1]
