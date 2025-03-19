# <<< DEPENDENCIES >>>
import matplotlib.pyplot as plt  # for plotting images
import numpy as np  # for numerical operations
from sklearn.model_selection import train_test_split  # for splitting the dataset
import torch  # for working with PyTorch
import torch.nn as nn  # for building neural networks
import torch.optim as optim  # for optimization algorithms
from skimage.metrics import structural_similarity as ssim  # for SSIM calculation

# <<< DATA LOADING & FEATURE ENGINEERING >>>
images2 = np.load("subset_2.npy")
images3 = np.load("subset_3.npy")

# Print shapes for debugging
print("subset_2.npy shape:", images2.shape)
print("subset_3.npy shape:", images3.shape)

# Normalize if images are in 0-255 range
images2 = images2 / 255.0
images3 = images3 / 255.0

images = np.load("subset_1.npy")  # loads the features from a numpy file
print("subset_1.npy shape:", images.shape)  # for debugging - checks files have been loaded in correctly
images = images / 255.0

# checks and reshapes if images are flattened - i encountered many shape-related images so this helped my process greatly
def process_images(np_images):
    if np_images.ndim == 2:  # flattened, shape (n, 101250)
        np_images = np_images.reshape(-1, 150, 225, 3)
        print("Reshaped flattened images to:", np_images.shape)
    elif np_images.ndim == 4 and np_images.shape[-1] == 3:
        print("Images are in HxWxC format:", np_images.shape)
    else:
        print("Unexpected image shape:", np_images.shape)
    return np_images

images = process_images(images)
images2 = process_images(images2)
images3 = process_images(images3)

# <<< MODEL DEVELOPMENT >>>
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # (batch, 16, 75, 113)
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (batch, 32, 38, 57)
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (batch, 64, 19, 29)
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (batch, 16, 75, 113)
            nn.ReLU(True),
        )
        # Decoder
        self.decoder = nn.Sequential(
          #  Upsample from (64,19,29) -> (64,38,58)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            # Upsample from (32,38,58) -> (32,76,116)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            # Upsample from (16,76,116) -> (16,152,232)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Ensures output values in [0,1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def run_all(np_images):
    # Split the data into training and testing sets
    labels = np.zeros((np_images.shape[0], 1))  # dummy labels
    trainX, testX, _, _ = train_test_split(np_images, labels, test_size=0.30)

    # Convert from HxWxC to CxHxW
    trainX = torch.from_numpy(trainX).to(torch.float).permute(0, 3, 1, 2)
    testX = torch.from_numpy(testX).to(torch.float).permute(0, 3, 1, 2)

    autoencoder = Autoencoder()  # instantiate autoencoder
    error = nn.MSELoss()  # loss function
    optimiser = optim.Adam(autoencoder.parameters(), lr=0.001)  # optimizer

    numberEpochs = 5  # number of epochs
    batchSize = 50  # batch size

    # Training loop
    for epoch in range(numberEpochs):
        epoch_loss = 0.0
        for num in range(0, trainX.shape[0], batchSize):
            inputs = trainX[num:num+batchSize]
            optimiser.zero_grad()
            outputs = autoencoder(inputs)
            outputs = torch.nn.functional.interpolate(outputs, size=(150, 225), mode='bilinear', align_corners=False)
            trainLoss = error(outputs, inputs)
            trainLoss.backward()
            optimiser.step()
            epoch_loss += trainLoss.item()
        print(f"Epoch {epoch+1}/{numberEpochs}, Loss: {epoch_loss:.4f}")

    # Model evaluation
    with torch.no_grad():
        testOutputs = autoencoder(testX)
        testOutputs = torch.nn.functional.interpolate(testOutputs, size=(150, 225), mode='bilinear', align_corners=False)
        testLoss = error(testOutputs, testX)
        print(f"Test Loss: {testLoss:.4f}")

    # Calculate compression ratio
    original_size = np.prod(trainX.shape[1:])  # size of the original data
    encoded_size = np.prod(autoencoder.encoder(trainX).shape[1:])  # size of the encoded data
    compression_ratio = original_size / encoded_size
    print(f"Compression Ratio: {compression_ratio:.4f}")

    # Calculate SSIM for each image
    ssim_values = []
    for i in range(testX.shape[0]):
        original_img = testX[i].permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC
        reconstructed_img = testOutputs[i].permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC
        ssim_value = ssim(original_img, reconstructed_img, data_range=1.0, channel_axis=2)
        ssim_values.append(ssim_value)
    avg_ssim = np.mean(ssim_values)
    print(f"Average SSIM: {avg_ssim:.4f}")

    # Figure creation: display images
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(5):
        # Original image (convert from CxHxW to HxWxC)
        original_img = np.transpose(testX[i].numpy(), (1, 2, 0))
        axs[0, i].imshow(original_img)
        axs[0, i].set_title("Original")
        axs[0, i].axis('off')

        # Reconstructed image
        reconstructed_img = np.transpose(testOutputs[i].numpy(), (1, 2, 0))
        axs[1, i].imshow(reconstructed_img)
        axs[1, i].set_title("Reconstructed")
        axs[1, i].axis('off')
    plt.show()

# Run on each subset
run_all(images)
run_all(images2)
run_all(images3)