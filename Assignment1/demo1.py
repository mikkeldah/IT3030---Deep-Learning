import matplotlib.pyplot as plt

from config import parse_file

from utils import *

# Network 1 (2 hidden layers):
nn = parse_file('Assignment1/configfiles/network1.txt')

# Data
features, targets, labels = get_doodler_data(count=5000)

X_train, y_train, X_test, y_test = train_test_split(features, targets, split=0.2)
X_train, y_train , X_val, y_val = train_test_split(X_train, y_train, split=0.2)

# Train Network on Data
nn.train(X_train, y_train, X_val, y_val)


# Test
test_losses = []
test_correct_preds = 0
for i in range(len(X_test)):
    x = X_test[i].flatten().reshape(-1, 1)
    y = y_test[i].reshape(1, -1)

    output = nn.forward_pass(x)
    loss = cross_entropy_loss(y, output)

    test_losses.append(loss)

    prediction = nn.predict(x)

    if np.array_equal(prediction.reshape(1, -1), y):
        test_correct_preds += 1

print("Test loss: ", sum(test_losses) / len(test_losses))
print("Test accuracy: ", str(int(100 * test_correct_preds / len(X_test)))+"%")


# Predict some test images
# ['ball', 'box', 'bar', 'triangle']
random_ints = np.random.randint(low=0, high=X_test.shape[0], size=2)

pred1 = nn.predict(X_test[random_ints[0]])
print(pred1)

pred2 = nn.predict(X_test[random_ints[1]])
print(pred2)


#subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(2,1) 

# use the created array to output your multiple images
axarr[0].imshow(X_test[random_ints[0]], cmap="gray")
axarr[1].imshow(X_test[random_ints[1]], cmap="gray")
 
plt.show()