import numpy as np
from doodler import *

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def softmax(output: np.ndarray) -> np.ndarray:
    return np.exp(output) / np.exp(output).sum(axis=0)

def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray) -> float:
    cross_entropy_loss = -np.sum(targets * np.log(outputs), axis=1)
    return cross_entropy_loss

def cross_entropy_loss_grad(targets: np.ndarray, outputs: np.ndarray) -> float:
    cross_entropy_loss_grad = -targets / outputs
    return cross_entropy_loss_grad


def train_test_split(features, targets, split=0.2):
    n_samples = features.shape[0]
    split = int(split * n_samples)
    indices = np.random.permutation(n_samples)

    train_indices = indices[split:]
    test_indices = indices[:split]

    X_train, y_train = features[train_indices], targets[train_indices]
    X_test, y_test = features[test_indices], targets[test_indices]

    return X_train, y_train, X_test, y_test


# types=['ball', 'box', 'bar', 'triangle']
def get_doodler_data(count):
    image_size = 28
    X_dood = gen_standard_cases(count=count, rows=image_size, cols=image_size, show=False, cent=True, types=['ball', 'box', 'bar', 'triangle'])
    features = X_dood[0]
    targets = X_dood[1]
    labels = X_dood[2]
    return features, targets, labels


def one_hot_encode(y: np.ndarray, n_classes: int):
    y_encoded = np.zeros((y.shape[0], n_classes))
    y_encoded[np.arange(y.shape[0]), y.flatten()] = 1

    return y_encoded





