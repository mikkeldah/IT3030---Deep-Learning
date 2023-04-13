import numpy as np

def to_one_hot(ndarray):
    num_classes = 10
    num_samples = ndarray.shape[0]
    one_hot = np.zeros((num_samples, num_classes))
    one_hot[np.arange(num_samples), ndarray] = 1
    return one_hot