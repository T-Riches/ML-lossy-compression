# <<< DEPENDENCIES >>>
import matplotlib.pyplot as plt  # for plotting images
import numpy as np  # for numerical operations
from sklearn.model_selection import train_test_split  # for splitting the dataset
import torch  # for working with PyTorch
import torch.nn as nn  # for building neural networks
import torch.optim as optim  # for optimization algorithms

# <<< DATA LOADING & FEATURE ENGINEERING >>>
images = np.load("subset_1.npy")  # load the features from a numpy file
images2 = np.load("subset_2.npy")  # load the features from a numpy file
images3 = np.load("subset_3.npy")  # load the features from a numpy file

images = images / 255  # normalize the pixel values to be between 0 and 1
images2 = images2 / 255
images3 = images3 / 255

# <<< MODEL DEVELOPMENT >>>
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # (batch_size, 16, 75, 113)
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (batch_size, 32, 38, 57)
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (batch_size, 64, 19, 29)
            nn.ReLU(True)
        )
        # Decoder
# Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),  # (batch_size, 32, 38, 57)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),  # (batch_size, 16, 75, 113)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),  # Fix output size
            nn.Sigmoid()  
        )



    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def run_all(images):
    # assuming all images in subset_1.npy belong to class 0
    labels = np.zeros((images.shape[0], 1))  # create an array of zeros with the same number of rows as images

    # split the data into training and testing sets
    trainX, testX, trainy, testy = train_test_split(images, labels, test_size=0.30)

    # convert the data to PyTorch tensors and reshape for convolutional layers
    trainX = torch.from_numpy(trainX).to(torch.float).view(-1, 3, 150, 225)  # Reshape input data
    testX = torch.from_numpy(testX).to(torch.float).view(-1, 3, 150, 225)  # Reshape input data
    trainy = torch.from_numpy(trainy).to(torch.float)
    testy = torch.from_numpy(testy).to(torch.float)

    autoencoder = ConvAutoencoder()  # instance of autoencoder

    error = nn.MSELoss()  # loss function
    optimiser = optim.Adam(autoencoder.parameters(), lr=0.001)  # Adam optimizer

    numberEpochs = 20  # number of times entire dataset is passed through the NN
    batchSize = 50  # batch size is the number of training samples that are fed to the neural network at once.

    for epoch in range(numberEpochs):
        loss = 0.0
        for num in range(0, trainX.shape[0], batchSize):
            inputs = trainX[num:num+batchSize]  # selects batch of data from trainX at num and ending at batch size
            optimiser.zero_grad()  # reset the gradients
            outputs = autoencoder(inputs)  # perform a forward pass
            outputs = torch.nn.functional.interpolate(outputs, size=(150, 225), mode='bilinear', align_corners=False)
            trainLoss = error(outputs, inputs)  # compute the loss
            trainLoss.backward()  # perform backpropagation
            optimiser.step()  # update the model parameters
            loss += trainLoss.item()  # accumulate the loss
        print(f"Epoch {epoch+1}/{numberEpochs}, Loss: {loss:.4f}")  # reminder for python: f"My name is {variable} and i am {variable2} years old"

    # <<< MODEL EVALUATION >>>
    with torch.no_grad():  # speeds stuff up bc it just does the forward passes without computing the gradients
        testOutputs = autoencoder(testX)
        testOutputs = torch.nn.functional.interpolate(testOutputs, size=(150, 225), mode='bilinear', align_corners=False)  # Resize output
        testLoss = error(testOutputs, testX)  # Compute loss
        print(f"Test Loss: {testLoss:.4f}")

    # <<< FIGURE CREATION >>>
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))

    for i in range(5):
        axs[0, i].imshow(np.transpose(testX[i].numpy(), (1, 2, 0)), cmap='gray')  # Correctly display the images
        axs[0, i].set_title("Original")
        axs[0, i].axis('off')

        # Reconstructed images
        axs[1, i].imshow(np.transpose(testOutputs[i].numpy(), (1, 2, 0)), cmap='gray')  # Correctly display the images
        axs[1, i].set_title("Reconstructed")
        axs[1, i].axis('off')

    plt.show()  # show the grid of images

run_all(images)
run_all(images2)
run_all(images3)