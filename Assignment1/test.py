import numpy as np
import matplotlib.pyplot as plt

from utils import *
from neural_network import NeuralNetwork
from enums import Activation, Cost
from layer import *
from doodler import *



features, targets, labels = get_doodler_data(count=10000)

X_train, y_train, X_test, y_test = train_test_split(features, targets, split=0.1)

#X_train, y_train, X_val, y_val = train_test_split(X_train, y_train, split=0.2)

# print(X_train.shape)
# print(y_train.shape)
# print(X_val.shape)
# print(y_val.shape)
# print(X_test.shape)
# print(y_test.shape)


nn = NeuralNetwork(cost_function=Cost.CROSS_ENTROPY_LOSS, lr=0.1)

nn.add_layer(DenseLayer(prev_layer_size=28*28, layer_size=100, activation=Activation.SIGMOID))
nn.add_layer(DenseLayer(prev_layer_size=100, layer_size=50, activation=Activation.SIGMOID))
nn.add_layer(DenseLayer(prev_layer_size=50, layer_size=20, activation=Activation.SIGMOID))
nn.add_layer(DenseLayer(prev_layer_size=20, layer_size=4, activation=Activation.SIGMOID))
nn.add_layer(SoftmaxLayer())


def train():

    training_scores = []
    losses = []
    losses_show = []

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

        if len(losses) > 20:
            losses_show.append(np.mean(losses[-20:]))
        else:
            losses_show.append(losses[-1])

        print("\n")

    plt.plot(range(len(losses_show)), losses_show)
    plt.show()

train()