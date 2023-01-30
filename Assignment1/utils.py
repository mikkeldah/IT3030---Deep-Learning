import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

def softmax(output: np.ndarray) -> np.ndarray:
    return np.exp(output) / np.exp(output).sum()

def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray) -> float:

    loss_vector = -(targets * np.log(outputs) + (1 - targets) * np.log(1 - outputs))
    cross_entropy_loss = (1 / targets.shape[0]) * np.sum(loss_vector)

    return cross_entropy_loss

def add_bias_dimension(x: np.array):
    biases = np.ones((1, x.shape[1]))
    return np.append(x, biases, axis=0)

def remove_bias_dimension(x: np.array):
    return x[:-1]

