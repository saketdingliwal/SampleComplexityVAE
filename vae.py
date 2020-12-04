import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
import scipy
import matplotlib.pyplot as plt


criterion = nn.MSELoss()
SIGMA = 1e-2




class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, d, D, size):
        v = np.random.normal(0, 1, size=(size,d))
        a = np.random.random(size=(D, d))
        self.A, _ = np.linalg.qr(a)
        self.sigma = SIGMA
        noise = np.random.normal(0, 1, size=(size,D))
        x_train = np.matmul(v, self.A.T) + self.sigma * noise
        x_train_tensor = torch.from_numpy(x_train).float()
        self.x = x_train_tensor
        
    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return len(self.x)




class Encoder(torch.nn.Module):
    def __init__(self, d, D):
        super(Encoder, self).__init__()
        self.M = torch.nn.Linear(D, d, bias=False)
        self.log_S = torch.nn.Parameter(torch.rand((d, 1)))

    def forward(self, x):
        return self.M(x)


class Decoder(torch.nn.Module):
    def __init__(self, d, D, set_s=None):
        super(Decoder, self).__init__()
        self.W = torch.nn.Linear(d, D, bias=False)
        if set_s is None:
            self.log_s = torch.nn.Parameter(torch.rand((1,)))
        else:
            self.log_s = torch.nn.Parameter(torch.log(torch.Tensor([set_s])), requires_grad=False)
        # print (self.log_s)

    def forward(self, x):
        return self.W(x)


class VAE(torch.nn.Module):

    def __init__(self, d, D, set_decoder_s=None):
        super(VAE, self).__init__()
        self.d = d
        self.D = D
        self.encoder = Encoder(d, D)
        self.decoder = Decoder(d, D, set_decoder_s)

    def _sample_latent(self, mu, sigma):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        std_z = torch.from_numpy(np.random.normal(0, 1, size=mu.size())).float()
        self.z_mean = mu
        self.z_sigma = sigma
        eps_sigm = sigma[None:] * Variable(torch.transpose(std_z,0,1), requires_grad=False)
        return mu +  torch.transpose(eps_sigm, 0, 1) # Reparameterization trick

    def forward(self, x):
        mu = self.encoder(x)
        # print (self.encoder.M.weight)
        # print (mu)
        sigma = torch.exp(self.encoder.log_S)
        z = self._sample_latent(mu, sigma)
        # print (sigma)
        # print (z)
        # print (self.decoder.W.weight)
        dec_mean = self.decoder(z)
        # print (dec_mean)
        return dec_mean
        # s = torch.exp(self.decoder.log_s)
        # std_z = torch.from_numpy(np.random.normal(0, 1, size=dec_mean.size())).float()
        # dec_sigma = s * Variable(std_z, requires_grad=False)
        # return dec_mean + dec_sigma

    def kl_loss(self):
        mean_sq = self.z_mean * self.z_mean
        log_term = torch.sum(torch.log(self.z_sigma))
        trace_term = torch.sum(self.z_sigma)    
        return 0.5 * torch.mean(trace_term + mean_sq - log_term - self.d)

    def re_loss(self, x, x_hat):
        s = torch.exp(self.decoder.log_s)
        mse_loss = criterion(x, x_hat)
        log_term = 2 * self.D * self.decoder.log_s + self. D * np.log(2*np.pi)
        return 0.5 * (mse_loss/(s*s) + log_term)


    def total_loss(self, x):
        x_hat = self.forward(x)
        return self.kl_loss() + self.re_loss(x, x_hat)

    def re_loss_direct(self, x):
        log_term = (self.D/2) * (np.log(2*np.pi) + 2*self.decoder.log_s)
        s = torch.exp(self.decoder.log_s)

        x_mat = torch.matmul(self.decoder.W.weight, self.encoder.M.weight) - torch.eye(self.D)
        x_vect = torch.matmul(x, x_mat)
        norm_term = torch.mean(x_vect * x_vect)/(2*s*s)
        # print (norm_term)

        W = self.decoder.W.weight
        W_T = torch.transpose(W, 0, 1)
        trace_mat = torch.matmul(W,W_T)
        trace_diag = torch.diagonal(trace_mat)
        S = torch.exp(self.encoder.log_S)
        trace_term = torch.sum(trace_diag * S)/(2*s*s)
        # print (log_term)
        # print (trace_term)

        return log_term + norm_term + trace_term


    def kl_loss_direct(self, x):
        S = torch.exp(self.encoder.log_S)
        mean = self.encoder(x)
        mean_term = torch.mean(mean * mean)
        trace_term = torch.sum(S)
        log_term = torch.sum(self.encoder.log_S)
        return 0.5 * (trace_term + mean_term - log_term - self.d)

    def total_loss_direct(self, x):
        return self.kl_loss_direct(x) + self.re_loss_direct(x)

    def avg_g(self, A, sigma, num_samples):
        W = self.decoder.W.weight.detach().numpy()
        mat = A - W
        z = np.random.normal(0, 1, size=(num_samples,self.d))
        diff_hat = np.matmul(z, mat.T)
        diff_hat_sq = diff_hat * diff_hat

        norm_term = np.mean(diff_hat_sq)/(sigma*sigma)
        return norm_term
        s = np.exp(self.decoder.log_s)
        ratio = s/sigma
        ratio_sq = ratio * ratio
        const_term = self.D * (ratio_sq - np.log(ratio_sq) -1)

        return 0.5*(norm_term + const_term)

        # const_term = 


    def avg_h(self, A, sigma, num_samples):
        v = np.random.normal(0, 1, size=(num_samples,self.d))
        noise = np.random.normal(0, 1, size=(num_samples,D))
        x = np.matmul(v, A.T) + sigma * noise
        log_S = self.encoder.log_S.detach().numpy()
        S = np.exp(log_S)
        P = A.T/(1+sigma*sigma)
        M = self.encoder.M.weight.detach().numpy()

        trace_term = (1+(1/(sigma*sigma)))*np.sum(S)
        log_term = self.d * np.log(sigma*sigma) - self.d * np.log(1 + sigma*sigma) - np.sum(log_S)
        mat_x = np.matmul(x, (P-M).T)
        mat_x_sq = np.sum(mat_x * mat_x)/num_samples
        mat_term = (1+(1/(sigma*sigma))) * mat_x_sq

        return 0.5 * (trace_term + mat_term + log_term - d)






if __name__ == '__main__':

    D = 10
    d = 5
    num_points = 10000
    batch_size = 100
    lr = 0.001
    max_epochs = 500
    dataset = MyDataSet(d, D, num_points)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    print('Number of samples: ', len(dataset))
    # print (dataset.A)
    # print (dataset.sigma)
    # s = 2 * np.sqrt(dataset.A * dataset.A + dataset.sigma * dataset.sigma) ## SC 
    s = dataset.sigma
    # print (s)
    vae = VAE(d, D, s)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    l = None
    epochs = []
    vae_losses = []
    g_losses = []
    h_losses = []
    for epoch in range(max_epochs):
        losses = []
        for i, data in enumerate(dataloader, 0):
            inputs = Variable(data)
            optimizer.zero_grad()
            loss = vae.total_loss_direct(inputs)
            loss.backward()
            optimizer.step()
            l = loss
            losses.append(l.data[0])
        print (epoch, "M", vae.encoder.M.weight)
        print (epoch, "W", vae.decoder.W.weight)
        print (epoch, "Log S",vae.encoder.log_S)
        print (epoch, "Log s",vae.decoder.log_s)
        vae_loss = np.mean(losses)
        g_loss = vae.avg_g(dataset.A, dataset.sigma, 10000)
        h_loss = vae.avg_h(dataset.A, dataset.sigma, 10000)
        epochs.append(epoch)
        vae_losses.append(vae_loss)
        g_losses.append(g_loss)
        h_losses.append(h_loss)
        print(epoch, vae_loss)
        print(epoch, g_loss)
        print(epoch, h_loss)
        print ("----------------")

    plt.plot(epochs, vae_losses, label="VAE Objective")
    # plt.plot(epochs, g_losses, label="E[g(W,s2)]")
    # plt.plot(epochs, h_losses, label="E[h(M,S)]")
    plt.legend()
    plt.savefig('run1.png')