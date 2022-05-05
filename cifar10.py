"""Neural Networks for CIFAR10 dataset"""

# Data and Libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as fun
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mnist import DNN, CNN, train, test

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classNames = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
accuracy_list = []


# Helper functions for training and testing:

def get_num_params(model):
    np = 0
    for p in list(model.parameters()):
        np += p.nelement()
    return np


def show_some_training_images(train_data):
    """function to show some training images"""
    plt.figure(figsize=(8, 3), dpi=200)
    image_batch, label_batch = next(iter(train_data))                   # fetch a batch of train images; RANDOM
    for i in range(20):
        image, label = image_batch[i], classNames[label_batch[i].item()]
        plt.subplot(2, 10, i+1)
        plt.imshow(np.transpose((image/2 + 0.5).numpy(), (1, 2, 0)))    # un-normalize and then convert from tensor
        plt.axis('off')
        plt.title(label)
    plt.show()


def visualize_perm(perm, train_data):
    """visualize a fixed permutation of the image pixels applied to all images"""
    plt.figure(figsize=(8, 8))                                          # show some training images
    image_batch, label_batch = next(iter(train_data))                   # fetch a batch of train images; RANDOM
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
    image_batch, label_batch = next(iter(test_data))                    # fetch a batch of test images
    if scramble:
        image_batch_scramble = image_batch.view(-1, 3072)[:, perm].view(-1, 3, 32, 32)
        with torch.no_grad():
            log_pred_prob_batch = model(image_batch_scramble)
        for i in range(10):
            img_perm, real_label = image_batch_scramble[i], label_batch[i].item()
            # Output of the network are log-probabilities, need to take exponential for probabilities
            pred_prob = torch.exp(log_pred_prob_batch[i]).data.numpy().squeeze()
            visualize_pred(img_perm, pred_prob, real_label)
    else:
        with torch.no_grad():
            log_pred_prob_batch = model(image_batch)
        for i in range(10):
            img, real_label = image_batch[i], label_batch[i].item()
            # Output of the network are log-probabilities, need to take exponential for probabilities
            pred_prob = torch.exp(log_pred_prob_batch[i]).data.numpy().squeeze()
            visualize_pred(img, pred_prob, real_label)
    plt.show()

# FROM: https://github.com/boazbk/mltheoryseminar/blob/main/code/hw0/simple_train.ipynb
# 5-Layer CNN for CIFAR
# This is the Myrtle5 network by David Page (https://myrtle.ai/learn/how-to-train-your-resnet-4-architecture/)


class Flatten(nn.Module):
    @staticmethod
    def forward(x):
        return x.view(x.size(0), x.size(1))


class PrintShape(nn.Module):
    @staticmethod
    def forward(x):
        print(x.shape)
        return x


def make_myrtle5(c=64, num_classes=10):
    """Returns a 5-layer CNN with width parameter c."""
    return nn.Sequential(
        # Layer 0
        nn.Conv2d(3, c, (3, 3), (1, 1), 1, bias=True), nn.BatchNorm2d(c), nn.ReLU(),
        # Layer 1
        nn.Conv2d(c, c*2, (3, 3), (1, 1), 1, bias=True), nn.BatchNorm2d(c*2), nn.ReLU(), nn.MaxPool2d(2),
        # Layer 2
        nn.Conv2d(c*2, c*4, (3, 3), (1, 1), 1, bias=True), nn.BatchNorm2d(c*4), nn.ReLU(), nn.MaxPool2d(2),
        # Layer 3
        nn.Conv2d(c*4, c*8, (3, 3), (1, 1), 1, bias=True), nn.BatchNorm2d(c*8), nn.ReLU(), nn.MaxPool2d(2),
        # Layer 4
        nn.MaxPool2d(4), Flatten(), nn.Linear(c*8, num_classes, bias=True), nn.LogSoftmax(dim=1)    # PrintShape(),
    )


def fgsm_attack(image, epsilon, data_grad):
    """FGSM attack function"""
    sign_data_grad = data_grad.sign()   # Collect the element-wise sign of the data gradient
    # Create the perturbed image by adjusting each pixel of the input image using the gradients
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, -1, 1)               # Adding clipping to maintain [0,1] range
    return perturbed_image                                              # Return the perturbed image


def fgsm_test(model, device, test_data, epsilon):
    correct, adv_examples, progress_count = 0, [], 0                    # Accuracy counter
    for data, target in test_data:      # Loop over all examples in test set, one by one (test_loader has batch_size=1)
        progress_count += 1
        data, target = data.to(device), target.to(device)               # Send the data and label to the device
        data.requires_grad = True       # Set requires_grad attribute of tensor. Important for Attack

        output = model(data)                                            # Forward pass the data through the model
        init_pred = output.max(1, keepdim=True)[1]                      # get the index of the max log-probability

        if init_pred.item() != target.item():
            continue    # If the model is wrong, then this can't be an adversarial example, move on to the next example

        loss = fun.nll_loss(output, target)                             # Calculate the loss
        model.zero_grad()                                               # Zero all existing gradients
        loss.backward()                                                 # Calculate gradients of model in backward pass
        data_grad = data.grad.data                                      # Collect gradients of the data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)          # Call FGSM Attack

        output = model(perturbed_data)                                  # Apply the model to the perturbed image
        final_pred = output.max(1, keepdim=True)[1]                     # get the index of the max log-probability

        if final_pred.item() != target.item():                  # check if the perturbation forces a mis-classification
            if len(adv_examples) < 10:                          # Save some adv examples for visualization later
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                real_im = data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex, real_im))
        else:
            correct += 1

        if progress_count % 1000 == 0:
            print(f'FGSM Attack Iteration: {progress_count * len(data)}')

    final_acc = correct / float(len(test_data))                         # Calculate final accuracy for this epsilon
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_data)} = {final_acc}")

    return final_acc, adv_examples                              # Return the accuracy and an adversarial example


def plot_acc_vs_eps(epsilons, accuracies):
    plt.figure(figsize=(5, 5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()


def plot_adv_samples(epsilons, adv_examples):
    """Plot several examples of adversarial samples at each epsilon"""
    plt.figure(figsize=(8, 20))
    for i in range(len(epsilons)):
        eps = epsilons[i]
        for j in range(len(adv_examples[i])):
            orig, adv, ex, img = adv_examples[i][j]
            noise = ex - img
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(figsize=(6, 9), ncols=4)
            ax1.imshow(np.transpose(img / 2 + 0.5, (1, 2, 0)))
            ax1.axis('off')
            ax1.set_title(classNames[orig])
            ax2.imshow(np.transpose(ex / 2 + 0.5, (1, 2, 0)))
            ax2.axis('off')
            ax2.set_title(classNames[adv])
            ax3.imshow(np.transpose((noise + 1)/ 2 + 0.5, (1, 2, 0)))
            ax3.axis('off')
            ax3.set_title("noise * {}".format(eps))
            ax4.imshow(np.transpose((noise / eps)/ 2 + 0.5, (1, 2, 0)))
            ax4.axis('off')
            ax4.set_title("noise")
            plt.tight_layout()


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on", device)        # this 'device' will be used for training our model

    train_loader, test_loader = DataLoader(                             # Load the CIFAR10 dataset dataset
        datasets.CIFAR10('./data', train=True, download=False,          # download=True if it's not already downloaded
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=64, shuffle=True), DataLoader(                       # shuffle=True randomizes the data
        datasets.CIFAR10('./data', train=False, download=False,         # download=True if it's not already downloaded
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=1, shuffle=True)                                     # batch_size=1000 for DNN, CNN, Larger CNN

    show_some_training_images(train_loader)

    fixed_perm = torch.randperm(3072)
    visualize_perm(fixed_perm, train_loader)

    # Train the Network: DNN or CNN
    # Model = CNN(input_size=3072, dataset_name='CIFAR10')
    Model = make_myrtle5()
    Model.to(device)
    optimizer = optim.SGD(Model.parameters(), lr=0.01, momentum=0.5)
    # print(f'Number of parameters: {Model.get_num_params()}')
    print(f'Number of parameters: {get_num_params(Model)}')

    for ep in range(0, 20):
        train(ep, Model, train_loader, fixed_perm, scramble=False, dataset_name='CIFAR10')
        test(Model, test_loader, fixed_perm, scramble=False, dataset_name='CIFAR10')

    Model.to('cpu')

    show_some_predictions(Model, test_loader, fixed_perm, scramble=False)

    epsilons = [0.01, 0.02, 0.04, 0.08, 0.16, .32]
    accuracies, adv_examples = [], []

    Model.to(device)

    for eps in epsilons:    # Run test for each epsilon
        acc, ex = fgsm_test(Model, device, test_loader, eps)
        accuracies.append(acc)
        adv_examples.append(ex)

    plot_acc_vs_eps(epsilons, accuracies)
    plot_adv_samples(epsilons, adv_examples)