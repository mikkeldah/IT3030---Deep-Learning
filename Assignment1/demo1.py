import matplotlib.pyplot as plt

from config import parse_file

from utils import *

# Network 1 (2 hidden layers):
nn = parse_file('Assignment1/configfiles/network1.txt')

# Data
features, targets, labels = get_doodler_data(count=5000)
X_train, y_train, X_test, y_test = train_test_split(features, targets, split=0.1)

# Train Network on Data
nn.train(X_train, y_train)

# Predict some test images
# ['ball', 'box', 'bar', 'triangle']
pred1 = nn.predict(X_test[1])
print(pred1)

pred2 = nn.predict(X_test[2])
print(pred2)



#subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(2,1) 

# use the created array to output your multiple images
axarr[0].imshow(X_test[1], cmap="gray")
axarr[1].imshow(X_test[2], cmap="gray")
 
plt.show()