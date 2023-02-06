from neural_network import *
from layer import *
from utils import *

def parse_file(filepath):
    # parse a config file 
    with open(filepath, 'r') as f:
        data = f.read()

    cost = str(data.split("\n")[1])
    lr = float(data.split("\n")[2])
    reglam = float(data.split("\n")[3])
    wrtype = str(data.split("\n")[4])

    layers = []

    for layer in data.split("\n")[6:]:
        name = layer.split(" ")[0]
        if name == "dense":
            win = layer.split(" ")[1].split(":")[1]
            wout = layer.split(" ")[2].split(":")[1]
            layers.append(DenseLayer(int(win), int(wout), activation=Activation.SIGMOID))
    
        elif name == "softmax":
            layers.append(SoftmaxLayer())

    nn = NeuralNetwork(cost_function=cost, lr=lr, reglam=reglam, wrtype=wrtype)
    nn.set_layers(layers=layers)

    return nn

