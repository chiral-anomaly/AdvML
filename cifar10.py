"""Neural Networks for CIFAR10 dataset"""

# Data and Libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mnist import DNN, CNN, train, test

classNames = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# Helper functions for training and testing:

def show_some_training_images(train_data):
    """function to show some training images"""
    plt.figure(figsize=(8, 3), dpi=200)
    image_batch, label_batch = next(iter(train_data))                # fetch a batch of train images; RANDOM
    for i in range(20):
        image, label = image_batch[i], classNames[label_batch[i].item()]
        plt.subplot(2, 10, i+1)
        plt.imshow(np.transpose((image/2 + 0.5).numpy(), (1, 2, 0)))    # un-normalize and then convert from tensor
        plt.axis('off')
        plt.title(label)
    plt.show()


def visualize_perm(perm, train_data):
    """visualize a fixed permutation of the image pixels applied to all images"""
    plt.figure(figsize=(8, 8))                                      # show some training images
    image_batch, label_batch = next(iter(train_data))             # fetch a batch of train images; RANDOM
    for i in range(6):
        image, label = image_batch[i], label_batch[i].item()
        image_perm = image.view(-1, 3072).clone()[:, perm].view(3, 32, 32)
        plt.subplot(3, 4, 2*i+1)
        plt.imshow(np.transpose((image/2 + 0.5).numpy(), (1, 2, 0)))
        plt.axis('off')
        plt.title(classNames[label])
        plt.subplot(3, 4, 2*i+2)
        plt.imshow(np.transpose((image_perm/2 + 0.5).numpy(), (1, 2, 0)))
        plt.axis('off')
        plt.title(classNames[label])
    plt.show()


def visualize_pred(img, pred_prob, real_label):
    """for viewing an image and its predicted classes"""
    _, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(np.transpose((img/2 + 0.5).numpy(), (1, 2, 0)))
    ax1.axis('off')
    pred_label = np.argmax(pred_prob)
    ax1.set_title([classNames[real_label], classNames[pred_label]])
    ax2.barh(np.arange(10), pred_prob)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(classNames)
    ax2.set_title('Prediction Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()


def show_some_predictions(model, test_data, perm, scramble: bool = False):
    """function to show some predictions on test images"""
    image_batch, label_batch = next(iter(test_data))    # fetch a batch of test images
    if scramble:
        image_batch_scramble = image_batch.view(-1, 3072)[:, perm].view(-1, 3, 32, 32)
        with torch.no_grad():   # Turn off gradients to speed up this part
            log_pred_prob_batch = model(image_batch_scramble)
        for i in range(10):
            img_perm, real_label = image_batch_scramble[i], label_batch[i].item()
            # Output of the network are log-probabilities, need to take exponential for probabilities
            pred_prob = torch.exp(log_pred_prob_batch[i]).data.numpy().squeeze()
            visualize_pred(img_perm, pred_prob, real_label)
    else:
        with torch.no_grad():   # Turn off gradients to speed up this part
            log_pred_prob_batch = model(image_batch)
        for i in range(10):
            img, real_label = image_batch[i], label_batch[i].item()
            # Output of the network are log-probabilities, need to take exponential for probabilities
            pred_prob = torch.exp(log_pred_prob_batch[i]).data.numpy().squeeze()
            visualize_pred(img, pred_prob, real_label)
    plt.show()


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on", device)    # this 'device' will be used for training our model

    train_loader, test_loader = DataLoader(                             # Load the CIFAR10 dataset dataset
        datasets.CIFAR10('./data', train=True, download=False,          # download=True if it's not already downloaded
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=64, shuffle=True), DataLoader(                       # shuffle=True randomizes the data
        datasets.CIFAR10('./data', train=False, download=False,         # download=True if it's not already downloaded
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=1000, shuffle=True)

    show_some_training_images(train_loader)

    fixed_perm = torch.randperm(3072)
    visualize_perm(fixed_perm, train_loader)

    # Train the Network: DNN or CNN
    Model = DNN(input_size=3072)

    Model.to(device)

    print(f'Number of parameters: {Model.get_num_params()}')

    for ep in range(0, 10):
        train(ep, Model, train_loader, fixed_perm, scramble=True, dataset_name='CIFAR10')
        test(Model, test_loader, fixed_perm, scramble=True, dataset_name='CIFAR10')

    Model.to('cpu')

    show_some_predictions(Model, test_loader, fixed_perm, scramble=True)
