"""Neural Networks for MNIST dataset"""

# Data and Libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as fun
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Helper functions for training and testing:

def show_some_training_images(training_data):
    """function to show some training images"""
    plt.figure(figsize=(16, 4))
    image_batch, label_batch = next(iter(training_data))            # fetch a batch of train images; RANDOM
    for i in range(20):
        image, label = image_batch[i], label_batch[i].item()
        plt.subplot(2, 10, i+1)
        plt.imshow(image.squeeze().numpy())
        plt.axis('off')
        plt.title(label)
    plt.show()


def visualize_perm(perm):
    """visualize a fixed permutation of the image pixels applied to all images"""
    plt.figure(figsize=(8, 8))                                      # show some training images
    image_batch, label_batch = next(iter(train_loader))             # fetch a batch of train images; RANDOM
    for i in range(6):
        image, label = image_batch[i], label_batch[i].item()
        image_perm = image.view(-1, 784).clone()[:, perm].view(-1, 1, 28, 28)
        plt.subplot(3, 4, 2*i+1)
        plt.imshow(image.squeeze().numpy())
        plt.axis('off')
        plt.title(label)
        plt.subplot(3, 4, 2*i+2)
        plt.imshow(image_perm.squeeze().numpy())
        plt.axis('off')
        plt.title(label)
    plt.show()


def train(epoch, model, perm=torch.arange(0, 784).long(), scramble: bool = False):
    """we pass a model object to this trainer, and it trains this model for one epoch"""
    model.train()                           # model in training mode. Turns on dropout, batch-norm etc during training
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)           # send to device
        if scramble:                                                # permute pixels
            data = data.view(-1, 784)[:, perm].view(-1, 1, 28, 28)
        optimizer.zero_grad()
        output = model(data)
        loss = fun.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def test(model, perm=torch.arange(0, 784).long(), scramble: bool = False):
    """for testing the model"""
    model.eval()                # model in evaluation mode. Turn off dropout, batch-norm etc during validation/testing
    test_loss, correct = 0, 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)           # send to device
        if scramble:    # permute pixels
            data = data.view(-1, 784)[:, perm].view(-1, 1, 28, 28)
        output = model(data)
        test_loss += fun.cross_entropy(output, target, reduction='sum').item()      # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]                  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    accuracy_list.append(accuracy)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({accuracy:.0f}%)\n')


def visualize_pred(img, pred_prob, real_label):
    """for viewing an image and its predicted classes"""
    _, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.numpy().squeeze())
    ax1.axis('off')
    pred_label = np.argmax(pred_prob)
    ax1.set_title([real_label, pred_label])
    ax2.barh(np.arange(10), pred_prob)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Prediction Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()


def show_some_predictions(test_data, perm=torch.arange(0, 784).long(), scramble: bool = False):
    """function to show some predictions on test images"""
    image_batch, label_batch = next(iter(test_data))                # fetch a batch of test images
    if scramble:
        image_batch_scramble = image_batch.view(-1, 784)[:, perm].view(-1, 1, 28, 28)
        with torch.no_grad():                                       # Turn off gradients to speed up this part
            log_pred_prob_batch = dnn(image_batch_scramble)
        for i in range(10):
            img_perm, real_label = image_batch_scramble[i], label_batch[i].item()
            # Output of the network are log-probabilities, need to take exponential for probabilities
            pred_prob = torch.exp(log_pred_prob_batch[i]).data.numpy().squeeze()
            visualize_pred(img_perm, pred_prob, real_label)
    else:
        with torch.no_grad():                                       # Turn off gradients to speed up this part
            log_pred_prob_batch = dnn(image_batch)
        for i in range(10):
            img, real_label = image_batch[i], label_batch[i].item()
            # Output of the network are log-probabilities, need to take exponential for probabilities
            pred_prob = torch.exp(log_pred_prob_batch[i]).data.numpy().squeeze()
            visualize_pred(img, pred_prob, real_label)
    plt.show()


class DNN(nn.Module):
    """Defines a Fully Connected Network with 2 Hidden Layers"""
    def __init__(self, input_size: int = 784, output_size: int = 10,    # images are 28x28 pixels; there are 10 classes
                 drop_out: bool = False, drop_rate: float = 0.25):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.drop_out = drop_out
        self.drop_rate = drop_rate
        if drop_out:                    # Network with Dropout Layers that randomly drop 25% neurons/connections out
            self.network = nn.Sequential(nn.Linear(input_size, 200), nn.Dropout(drop_rate), nn.ReLU(),
                                         nn.Linear(200, 100), nn.Dropout(drop_rate), nn.ReLU(),
                                         nn.Linear(100, 60), nn.ReLU(),
                                         nn.Linear(60, output_size), nn.LogSoftmax(dim=1))
        else:
            self.network = nn.Sequential(
                nn.Linear(input_size, 200), nn.ReLU(),              # input layer, ReLU activation function
                nn.Linear(200, 100), nn.ReLU(),                     # first hidden layer, ReLU activation function
                nn.Linear(100, 60), nn.ReLU(),                      # second hidden layer, ReLU activation function
                nn.Linear(60, output_size), nn.LogSoftmax(dim=1))   # output layer, log(Softmax(x)) function

    def forward(self, xyz):
        """forward pass"""
        xyz = xyz.view(-1, self.input_size)
        return self.network(xyz)

    def get_n_params(self):
        """method to count the number of parameters"""
        num_params = 0
        for param in list(self.parameters()):
            num_params += param.nelement()
        return num_params


class CNN(DNN):
    """The Convolutional Network"""
    def __init__(self, input_size: int = 784, output_size: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, (3,3))    # (in_channels, out_channels, kernel_size) respectively
        self.conv2 = nn.Conv2d(12, 24, (6,6))
        self.conv3 = nn.Conv2d(24, 32, (6,6))
        self.fc1 = nn.Linear(128, 200)      # 8*4*4=128
        self.fc2 = nn.Linear(200, 10)

    def forward(self, xyz, verbose=False):
        xyz = self.conv1(xyz)
        xyz = fun.relu(xyz)
        xyz = self.conv2(xyz)
        xyz = fun.relu(xyz)
        xyz = fun.max_pool2d(xyz, kernel_size=2)
        xyz = self.conv3(xyz)
        xyz = fun.relu(xyz)
        x = fun.max_pool2d(x, kernel_size=2)
        x = x.view(-1, 128)     # 8*4*4=128
        x = self.fc1(x)
        x = fun.relu(x)
        x = self.fc2(x)
        x = fun.log_softmax(x, dim=1)


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on", device)                                    # this 'device' will be used for training our model

    train_loader, test_loader = DataLoader(                         # Load the MNIST dataset
        datasets.MNIST('./data', train=True, download=False,
                       transform=transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=64, shuffle=True), DataLoader(                   # shuffle=True randomizes the data
        datasets.MNIST('./data', train=False, download=False,
                       transform=transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=1000, shuffle=True)

    show_some_training_images(train_loader)

    fixed_perm = torch.randperm(784)                                # fix a permutation; no need for this
    visualize_perm(fixed_perm)                                      # no need for this when scramble=False

    dnn = DNN(drop_out=True)                                        # set drop_out=False to switch the dropout off
    dnn.to(device)
    print(f'Number of parameters: {dnn.get_n_params()}')
    optimizer = optim.SGD(dnn.parameters(), lr=0.01, momentum=0.5)
    accuracy_list = []
    for ep in range(0, 10):                                         # Train the Network for 10 epochs
        train(ep, dnn, fixed_perm, scramble=True)                   # train(ep, dnn) for no scrambling
        test(dnn, fixed_perm, scramble=True)                        # test(dnn) for no scrambling

    dnn.to('cpu')
    show_some_predictions(test_loader, fixed_perm, scramble=True)   # Show some predictions of the test network
