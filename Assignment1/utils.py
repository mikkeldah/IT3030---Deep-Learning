import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

def softmax(output: np.ndarray) -> np.ndarray:
    print(np.exp(output))
    print(sum(np.exp(output)))
    return np.exp(output) / np.exp(output).sum(axis=0)

def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray) -> float:
    cross_entropy_loss = np.mean(-np.sum(targets * np.log(outputs), axis=1))
    return cross_entropy_loss

def cross_entropy_loss_grad(targets: np.ndarray, outputs: np.ndarray) -> float:
    cross_entropy_loss_grad = np.mean(-np.sum(targets / outputs, axis=1))
    return cross_entropy_loss_grad

def add_bias_dimension(x: np.array):
    biases = np.ones((1, x.shape[1]))
    return np.append(x, biases, axis=0)

def remove_bias_dimension(x: np.array):
    return x[:-1]

def one_hot_encode(y: np.ndarray, n_classes: int):
    y_encoded = np.zeros((y.shape[0], n_classes))
    y_encoded[np.arange(y.shape[0]), y.flatten()] = 1

    return y_encoded


