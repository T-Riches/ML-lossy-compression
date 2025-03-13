# <<< DEPENDENCIES >>>
import matplotlib.pyplot as plt  # for plotting images
import numpy as np  # for numerical operations
from sklearn.preprocessing import OneHotEncoder  # for one-hot encoding labels
from sklearn.model_selection import train_test_split  # for splitting the dataset
from sklearn.metrics import accuracy_score  # for evaluating the model's accuracy
import torch  # for working with PyTorch
import torch.nn as nn  # for building neural networks
import torch.nn.functional as F  # for activation functions
import torch.optim as optim  # for optimization algorithms

# <<< DATA LOADING & FEATURE ENGINEERING >>>
images = np.load("features.npy")  # load the features from a numpy file
images = images / 255  # normalize the pixel values to be between 0 and 1
labels = np.load("labels.npy")  # load the labels from a numpy file
labels = OneHotEncoder(sparse_output=False).fit_transform(labels.reshape(-1, 1))  # one-hot encode the labels
train_X, test_X, train_y, test_y = train_test_split(images, labels, test_size=0.30)  # split the data into training and testing sets
train_X = torch.from_numpy(train_X).to(torch.float)  # convert training features to a PyTorch tensor
train_y = torch.from_numpy(train_y).to(torch.float)  # convert training labels to a PyTorch tensor
test_X = torch.from_numpy(test_X).to(torch.float)  # convert testing features to a PyTorch tensor
test_y = torch.from_numpy(test_y).to(torch.float)  # convert testing labels to a PyTorch tensor

# <<< MODEL DEVELOPMENT >>>
class Net(nn.Module):  # define a neural network class that inherits from nn.Module
    def __init__(self):  # initialize the network
        super().__init__()  # call the parent class's initializer
        self.hidden_layers = [  # define the hidden layers
            nn.Linear(101250, 200),  # first hidden layer with 101250 input features and 200 output features
            nn.Linear(200, 10)  # second hidden layer with 200 input features and 10 output features
        ]
        self.output_layer = nn.Linear(10, 4)  # define the output layer with 10 input features and 4 output features

    def forward(self, x):  # define the forward pass
        for layer in self.hidden_layers:  # iterate over the hidden layers
            x = F.relu(layer(x))  # apply ReLU activation function to each hidden layer
        return F.softmax(F.sigmoid(self.output_layer(x)))  # apply sigmoid to the output layer and then softmax

my_network = Net()  # create an instance of the network
criterion = nn.BCELoss()  # define the loss function as binary cross-entropy loss
optimizer = optim.Adam(my_network.parameters(), learning_rate=0.01)  # define the optimizer as Adam with a learning rate of 0.01

# <<< MODEL EVALUATION >>>
batch_size = 50  # set the batch size
for epoch in range(5):  # train for 5 epochs
    loss = 0.0  # initialize the loss for the epoch
    for idx in range(0, train_X.shape[0], batch_size):  # iterate over the training data in batches
        optimizer.zero_grad()  # reset the gradients
        outputs = my_network(train_X[idx:idx+batch_size])  # perform a forward pass
        new_loss = criterion(outputs, train_y[idx:idx+batch_size])  # compute the loss
        new_loss.backward()  # perform backpropagation
        optimizer.step()  # update the model parameters
        loss = loss + new_loss.item()  # accumulate the loss
    print(f"Epoch {epoch} loss: {loss:.3f}")  # print the loss for the epoch

    test_classes = my_network(test_X)  # perform a forward pass on the test data
    test_classes = test_classes.argmax(axis=1)  # get the predicted classes
    print(f"Test Accuracy: {accuracy_score(test_classes, test_y.argmax(axis=1))}")  # compute and print the accuracy

# <<< FIGURE CREATION >>>
one_image = np.reshape(images[2, :], (150, 225, 3))  # reshape one image for display
plt.imshow(one_image, cmap='grey')  # display the image
fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(7.5, 5))  # create a 5x5 grid of subplots

for i in range(5):  # iterate over the rows
    for j in range(5):  # iterate over the columns
        axs[i, j].imshow(np.reshape(images[(i*10)+j, :], (25, 50)), cmap='grey')  # display each image in the grid
        axs[i, j].set_xticks([])  # remove x-axis ticks
        axs[i, j].set_yticks([])  # remove y-axis ticks