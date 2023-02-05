import numpy as np
import matplotlib.pyplot as plt

from utils import *
from neural_network import NeuralNetwork
from enums import Activation, Cost
from layer import *
from doodler import *



image_size = 28
X_dood = gen_standard_cases(count=1000, rows=image_size, cols=image_size, show=False, cent=True, types=['ball', 'box', 'bar', 'triangle'])
features = X_dood[0]
targets = X_dood[1]
labels = X_dood[2]

X_train, y_train, X_test, y_test = train_test_split(features, targets, split=0.1)

X_train, y_train, X_val, y_val = train_test_split(X_train, y_train, split=0.2)

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
print(X_test.shape)
print(y_test.shape)


nn = NeuralNetwork(cost_function=Cost.CROSS_ENTROPY_LOSS, lr=0.1)


nn.add_layer(DenseLayer(prev_layer_size=image_size*image_size, layer_size=100, activation=Activation.SIGMOID))
nn.add_layer(DenseLayer(prev_layer_size=100, layer_size=20, activation=Activation.SIGMOID))
nn.add_layer(DenseLayer(prev_layer_size=20, layer_size=4, activation=Activation.SIGMOID))
nn.add_layer(SoftmaxLayer())


def train():

    training_scores = []
    losses = []

    # For each training example repeat this process:
    for i in range(features.shape[0]):

        print("Training Case: ", i+1)

        # Preprosess data
        x = features[i].flatten().reshape(-1, 1)
        y = targets[i].reshape(1,-1)

        # print(y)
        # plt.imshow(features[i], cmap="gray")
        # plt.show()

        # Forward pass - output.shape is [batch size, num_classes]
        output = nn.forward_pass(x)
        print("Output Forward Pass: ", output)
        print("Target Value: ", targets[i].reshape(1, -1))

        # Backward pass
        nn.backward_pass(output, y)

        # Validation and Loss
        loss = cross_entropy_loss(y, output)
        print("Loss: ", loss)

        losses.append(loss)
        print("\n")

    plt.plot(range(len(losses)), losses)
    plt.show()

train()