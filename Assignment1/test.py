import numpy as np
import matplotlib.pyplot as plt

from utils import *
from neural_network import NeuralNetwork
from enums import Activation, Cost
from layer import *
from doodler import *



image_size = 30
X_dood = gen_standard_cases(count=50000, rows=image_size, cols=image_size, show=False, cent=True, types=['ball', 'box', 'bar', 'triangle'])
features = X_dood[0]
targets = X_dood[1]
labels = X_dood[2]

X_train, y_train, X_test, y_test = train_test_split(features, targets)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


nn = NeuralNetwork(cost_function=Cost.CROSS_ENTROPY_LOSS)


nn.add_layer(DenseLayer(prev_layer_size=image_size*image_size, layer_size=20, activation=Activation.SIGMOID))
nn.add_layer(DenseLayer(prev_layer_size=20, layer_size=4, activation=Activation.SIGMOID))
nn.add_layer(SoftmaxLayer())


def train():

    lr = 0.1

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

        # Backward pass
        nn.backward_pass(output, y, lr=lr)

        # Validation and Loss
        loss = cross_entropy_loss(y, output)
        print("Loss: ", loss)

        losses.append(loss)
        print("\n")

    plt.plot(range(len(losses)), losses)
    plt.show()

train()