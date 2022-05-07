import torch
from torch import nn
import torch.nn.functional as fun
from torchvision import transforms, datasets
from matplotlib import pyplot as plt


# Utility functions

def to_img(x):
    """Convert vector to image"""
    x = 0.5 * (x + 1)
    x = x.view(x.size(0), 28, 28)
    return x


def display_images(in_raw, out_raw, n=1):
    """Display images, n =m ax number of images to show"""
    out_raw = out_raw[:n]
    if in_raw is not None:
        in_raw = in_raw[:n]
        in_pic = to_img(in_raw.cpu().data).view(-1, 28, 28)
        plt.figure(figsize=(18, 6))
        for i in range(n):
            plt.subplot(1, n, i+1)
            plt.imshow(in_pic[i])
            plt.axis('off')
    out_pic = to_img(out_raw.cpu().data).view(-1, 28, 28)
    plt.figure(figsize=(18, 6))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(out_pic[i])
        plt.axis('off')
    plt.show()


class Autoencoder(nn.Module):
    """Simple autoencoder: Define model architecture"""
    def __init__(self, input_size=784, hidden_size=50, code_size=50):
        super().__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_size
        self.code_size = code_size  # latent space dimension or size of the hidden state
        self.encoder = nn.Sequential(
            # nn.Linear(input_size, hidden_size),
            # nn.ReLU(),
            # nn.Linear(hidden_size, code_size),
            nn.Linear(input_size, code_size), nn.Tanh()
        )
        self.decoder = nn.Sequential(
            # nn.Linear(code_size, hidden_size),
            # nn.ReLU(),
            # nn.Linear(hidden_size, input_size),
            nn.Linear(code_size, input_size), nn.Tanh()
        )

    def forward(self, x):
        h = self.encoder(x)
        x = self.decoder(h)
        return x, h

    def generate(self, n=10):
        z = torch.randn((n, self.code_size)).to(device)
        gen_img = self.decoder(z)
        return gen_img


class VAE(nn.Module):
    def __init__(self, d=20):
        super().__init__()
        self.d = d  # latent dimension
        self.encoder = nn.Sequential(nn.Linear(784, d ** 2), nn.ReLU(), nn.Linear(d ** 2, d * 2))
        self.decoder = nn.Sequential(nn.Linear(d, d ** 2), nn.ReLU(), nn.Linear(d ** 2, 784), nn.Tanh())

    def sampler(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new_empty(std.size()).normal_()
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu_logvar = self.encoder(x.view(-1, 784)).view(-1, 2, self.d)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.sampler(mu, logvar)
        return self.decoder(z), mu, logvar

    def generate(self, N=10):
        z = torch.randn((N, self.d)).to(device)
        gen_img = self.decoder(z)
        return gen_img


def loss_function(x_hat, x, mu, logvar, beta=1):
    recon_loss = fun.mse_loss(x_hat, x.view(-1, 784), reduction='sum')
    KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))
    return recon_loss + beta * KLD


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Data loader functions, define data loading step
    img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_loader, test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=False, transform=img_transform),
        batch_size=256, shuffle=True), torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=img_transform),
        batch_size=32, shuffle=True)

    model_1 = Autoencoder().to(device)  # Define the Autoencoder
    optimizer = torch.optim.Adam(model_1.parameters(), lr=1e-3)   # Configure the optimiser
    loss_fn = nn.MSELoss()              # loss function

    # Training the Autoencoder
    model_1.train()
    num_epochs = 10
    for epoch in range(num_epochs):
        for data in train_loader:
            img, _ = data
            img = img.to(device)
            img = img.view(img.size(0), -1)

            # forward step
            output, h = model_1(img)
            loss = loss_fn(output, img.data)

            # backward step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch == 0 or (epoch + 1) % 5 == 0:      # print some statistics
            print(f'epoch [{epoch + 1}/{num_epochs}], loss:{loss.item():.4f}')
            display_images(None, output, 4)

    # Test: Tryout some examples from the Test Set
    model_1.eval()
    img, _ = next(iter(test_loader))
    img = img.to(device)
    img = img.view(img.size(0), -1)
    output, h = model_1(img)
    loss = loss_fn(output, img.data)
    # =================== log ========================
    print(f'loss:{loss.item():.4f}')
    display_images(img, output, 10)

    N = 5   # Generating a few images from randomly sampled z
    display_images(None, model_1.generate(N), N)

    """ Denoising Autoencoder
    This uses the same model and loss function as before, but the training process adds noise to the input.
    The noise is generated using a normal distribution. """

    # define the model
    model_2 = Autoencoder(code_size=100).to(device)
    optimizer = torch.optim.Adam(model_2.parameters(), lr=1e-3)     # Configure the optimiser

    loss_fn = nn.MSELoss()      # loss function

    # Train
    model_2.train()
    num_epochs = 10
    do = nn.Dropout(p=0.5)
    for epoch in range(num_epochs):
        for data in train_loader:
            img, _ = data
            img = img.to(device)
            img = img.view(img.size(0), -1)
            noise = do(torch.ones(img.shape)).to(device)
            img_bad = (img * noise).to(device)
            # =================== forward =====================
            output, h = model_2(img_bad)
            loss = loss_fn(output, img.data)  # the reconstruction loss is still wrt the normal image
            # =================== backward ====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print some statistics
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f'epoch [{epoch + 1}/{num_epochs}], loss:{loss.item():.4f}')
            display_images(img_bad, output, 4)

    # Test images
    model_2.eval()
    img, _ = next(iter(test_loader))
    img = img.to(device)
    img = img.view(img.size(0), -1)
    do = nn.Dropout(p=0.5)
    noise = do(torch.ones(img.shape)).to(device)
    img_bad = ((img * noise) / 2).to(device)
    output, h = model_2(img_bad)
    loss = loss_fn(output, img.data)
    # =================== log ========================
    print(f'loss:{loss.item():.4f}')
    display_images(img, img_bad, 10)
    display_images(None, output, 10)

    """ Compare Simple Autoencoder with Denoising Autoencoder
    both are fed noisy inputs pulled from the test set """

    # Test
    model_1.eval()  # simple AE
    model_2.eval()  # denoising AW
    img, _ = next(iter(test_loader))
    img = img.to(device)
    img = img.view(img.size(0), -1)

    do = nn.Dropout(p=0.5)
    noise = do(torch.ones(img.shape)).to(device)

    img_bad = (img * noise).to(device)
    output_1, _ = model_1(img_bad)
    output_2, _ = model_2(img_bad)

    print("Noisy Images")
    display_images(None, img_bad, 10)
    print("Reconstruction using simple autoencoder")
    display_images(None, output_1, 10)
    print("Reconstruction using denoising autoencoder")
    display_images(None, output_2, 10)

    # Train the VAE
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,)  # Setting the optimiser
    epochs = 20
    codes = dict(mu=list(), logσ2=list(), y=list())
    for epoch in range(0, epochs + 1):
        # Training
        if epoch > 0:  # test untrained net first
            model.train()
            train_loss = 0
            for x, _ in train_loader:
                x = x.to(device)
                # =================== forward =====================
                x_hat, mu, logvar = model(x)
                loss = loss_function(x_hat, x, mu, logvar)
                train_loss += loss.item()
                # =================== backward ====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # =================== log ========================
            print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

        # Testing
        means, logvars, labels = list(), list(), list()
        with torch.no_grad():
            model.eval()
            test_loss = 0
            for x, y in test_loader:
                x = x.to(device)
                # =================== forward =====================
                x_hat, mu, logvar = model(x)
                test_loss += loss_function(x_hat, x, mu, logvar).item()
                # ===================== log =======================
                means.append(mu.detach())
                logvars.append(logvar.detach())
                labels.append(y.detach())
        # =================== log ========================
        codes['mu'].append(torch.cat(means))
        codes['logσ2'].append(torch.cat(logvars))
        codes['y'].append(torch.cat(labels))
        test_loss /= len(test_loader.dataset)
        print(f'====> Test set loss: {test_loss:.4f}')
        display_images(x, x_hat, 4)

    model.to(device)    # optional, depending on what was executed before

    # Generating a few images from randomly sampled z
    N = 6
    display_images(None, model.generate(N), N)
    display_images(None, model.generate(N), N)

    model.to('cpu')     # without this the code below gives RuntimeError

    # Interpolating between two images
    x, _ = next(iter(test_loader))
    x_hat, mu, logvar = model(x)
    A, B = 1, 14
    sample = model.decoder(torch.stack((mu[A].data, mu[B].data), 0))
    display_images(None, torch.stack((x[A].data.view(-1), x[B].data.view(-1)), 0), 2)
    display_images(None, torch.stack((sample.data[0], sample.data[1]), 0), 2)

    model.to(device)    # this is again important to add to avoid RuntimeError for the code below

    # Perform an interpolation between input A and B, in N steps
    N = 20
    code = torch.Tensor(N, 20).to(device)
    sample = torch.Tensor(N, 28, 28).to(device)
    for i in range(N):
        code[i] = i / (N - 1) * mu[B].data + (1 - i / (N - 1)) * mu[A].data
        sample = model.decoder(code)
    for i in range(N // 4):
        display_images(None, sample[i * 4:4 * (i + 1)], 4)
