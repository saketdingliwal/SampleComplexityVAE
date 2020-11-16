import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt


class Normal(object):
    def __init__(self, mu, sigma, log_sigma, v=None, r=None):
        self.mu = mu
        self.sigma = sigma  # either stdev diagonal itself, or stdev diagonal from decomposition
        self.logsigma = log_sigma
        dim = mu.get_shape()
        if v is None:
            v = torch.FloatTensor(*dim)
        if r is None:
            r = torch.FloatTensor(*dim)
        self.v = v
        self.r = r


class Encoder(torch.nn.Module):
    def __init__(self, d, D):
        super(Encoder, self).__init__()
        self.M = torch.nn.Linear(D, d, bias=False)
        self.log_S = torch.rand((d, 1), requires_grad=True)

    def forward(self, x):
        return self.M(x)


class Decoder(torch.nn.Module):
    def __init__(self, d, D):
        super(Decoder, self).__init__()
        self.W = torch.nn.Linear(d, D, bias=False)
        self.log_s = torch.rand(1, requires_grad= True)

    def forward(self, x):
        return self.W(x)


class VAE(torch.nn.Module):
    latent_dim = 8

    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # self._enc_mu = torch.nn.Linear(100, 8)
        # self._enc_log_sigma = torch.nn.Linear(100, 8)

    def _sample_latent(self, mu, sigma):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, state):
        mu = self.encoder(state)
        sigma = torch.exp(self.encoder.log_S)

        z = self._sample_latent(mu, sigma)

        dec_mean = self.decoder(z)
        s = torch.exp(self.decoder.log_s)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=dec_mean.size())).float()
        dec_sigma = s * Variable(std_z, requires_grad=False)
        return dec_mean + dec_sigma


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


if __name__ == '__main__':

    input_dim = 28 * 28
    batch_size = 32

    transform = transforms.Compose(
        [transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)

    dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    print('Number of samples: ', len(mnist))

    encoder = Encoder(input_dim, 100, 100)
    decoder = Decoder(8, 100, input_dim)
    vae = VAE(encoder, decoder)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    l = None
    for epoch in range(100):
        for i, data in enumerate(dataloader, 0):
            inputs, classes = data
            inputs, classes = Variable(inputs.resize_(batch_size, input_dim)), Variable(classes)
            optimizer.zero_grad()
            dec = vae(inputs)
            ll = latent_loss(vae.z_mean, vae.z_sigma)
            loss = criterion(dec, inputs) + ll
            loss.backward()
            optimizer.step()
            l = loss.data[0]
        print(epoch, l)

    plt.imshow(vae(inputs).data[0].numpy().reshape(28, 28), cmap='gray')
    plt.show(block=True)