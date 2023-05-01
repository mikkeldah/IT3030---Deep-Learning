import numpy as np
import matplotlib.pyplot as plt

from stacked_mnist import StackedMNISTData, DataMode

from models.mnist_classifier import MNISTClassifier

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Using the provided mnist dataset class
gen = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=9)

train_data = gen.get_full_data_set(training=True)
test_data = gen.get_full_data_set(training=False)


x_train, y_train = train_data
x_test, y_test = test_data

# Convert y_train and y_test into one-hot vectors
y_train = to_one_hot(y_train)
y_test = to_one_hot(y_test)

img = x_train[np.random.randint(0, x_train.shape[0])]


# Create pytorch dataloaders from x_train and y_train
x_train = torch.from_numpy(x_train.astype(np.float32)).permute(0, 3, 1, 2)
y_train = torch.from_numpy(y_train.astype(np.float32))

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

# Create pytorch dataloaders from x_test and y_test
x_test = torch.from_numpy(x_test.astype(np.float32)).permute(0, 3, 1, 2)
y_test = torch.from_numpy(y_test.astype(np.float32))

test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=32)



# Training setup
model = MNISTClassifier(image_depth=1)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()

# Training loop

for epoch in range(30):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch} loss: {total_loss}")

# Save the model
torch.save(model.state_dict(), "Project 3 - Deep Generative Models/trained_models/mnist_model.pt")


model.eval()
with torch.no_grad():
    correct_preds = 0
    test_loss = 0
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        test_loss += loss.item()

        pred = pred.argmax(dim=1, keepdim=True)
        truth = y.argmax(dim=1, keepdim=True)
        correct_preds += pred.eq(truth.view_as(pred)).sum().item()
        

    test_accuracy = correct_preds / len(test_loader.dataset)

print(f"Test loss: {test_loss}")
print(f"Test accuracy: {100*test_accuracy}%")