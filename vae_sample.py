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
import * from vae


if __name__ == '__main__':

    adjusted = 1
    s_trainable = 0
    Non = 'Non-trainable > A^2 + sigma^2 '
    goal = 'Gold'
    if adjusted:
        goal = 'Adjusted'
    if s_trainable:
        Non = 'Trainable'

    D = 1
    d = 1
    lr = 0.1
    max_epochs = 1000
    thresh = 1e-5
    num_points = 10000
    dataset = MyDataSet(d, D, num_points)
    # ns = [50, 100, 200, 500, 1000, 5000, 10000]
    ns = [50, 100, 250, 500, 1000, 5000, 10000, 50000]
    # ns = [500, 1000]
    max_runs = 40
    vae_losses = []
    g_losses = []
    h_losses = []
    for n in ns:
        vae_run = []
        g_run = []
        h_run = []
        for run in range(max_runs):
            batch_size = min(n, 1000)
            dataset.change_size(n)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                     shuffle=True, num_workers=2)
            print('Number of samples: ', len(dataset))
            # print (dataset.A)
            # print (dataset.sigma)
            # s = 2 * np.sqrt(dataset.A[0][0] * dataset.A[0][0] + dataset.sigma * dataset.sigma) ## SC 
            s = dataset.sigma
            # print (s)
            if s_trainable:
                vae = VAE(d, D)
            else:
                vae = VAE(d, D, s)
            optimizer = optim.Adam(vae.parameters(), lr=lr)
            l = None
            epochs = []
            prev_loss = 0
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
                # print (n, epoch, "A", dataset.A[0][0])
                # print (n, epoch, "M", vae.encoder.M.weight)
                # print (n, epoch, "W", vae.decoder.W.weight)
                # print (n, epoch, "Log S",vae.encoder.log_S)
                # print (n, epoch, "Log s",vae.decoder.log_s)
                vae_loss = np.mean(losses)
                if abs(vae_loss-prev_loss) < thresh:
                    break
                prev_loss = vae_loss

            print ("Training Done")
            g_loss = vae.expected_g(dataset.A, dataset.sigma, adjusted)
            h_loss = vae.expected_h(dataset.A, dataset.sigma, adjusted)

            vae_run.append(vae_loss)
            g_run.append(g_loss)
            h_run.append(h_loss)

            print(epoch, vae_loss)
            print(epoch, g_loss)
            print(epoch, h_loss)
            print ("----------------")

        print (n)
        print (vae_run)
        print (g_run)
        print (h_run)
        vae_losses.append(vae_run)
        g_losses.append(g_run)
        h_losses.append(h_run)

    print (vae_losses)
    print (g_losses)
    print (h_losses)

    with open('vae_2.npy', 'wb') as f:
        vae_loss_n = np.asarray(vae_losses)
        np.save(f, vae_loss_n)

    with open('g_2.npy', 'wb') as f:
        g_loss_n = np.asarray(g_losses)
        np.save(f, g_loss_n)

    with open('h_2.npy', 'wb') as f:
        h_loss_n = np.asarray(h_losses)
        np.save(f, h_loss_n)

    exit(0)

    fig, ax1 = plt.subplots(1,2)
    goals = ['Encoder {} Goal'.format(goal), 'Decoder {} Goal'.format(goal)]
    vals = [h_losses, g_losses]

    for i in range(2):
        loss_color = 'green'
        ax1[i].set_xlabel('number of samples')
        ax1[i].set_xscale('log')
        ax1[i].set_ylabel('VAE loss', color=loss_color)
        ax1[i].plot(ns, vae_losses, color=loss_color)
        ax1[i].tick_params(axis='y', labelcolor=loss_color)

        ax2 = ax1[i].twinx()  # instantiate a second axes that shares the same x-axis

        goal_color = 'blue'

        ax2.set_ylabel(goals[i], color=goal_color)  # we already handled the x-label with ax1
        ax2.plot(ns, vals[i], color=goal_color)
        ax2.tick_params(axis='y', labelcolor=goal_color)


    sigma_2 = dataset.sigma * dataset.sigma
    sigma_val = str(round(sigma_2,4))
    if s_trainable:
        s_val = str(round(torch.exp(vae.decoder.log_s).detach().numpy()[0],3))
    else:
        s_val = str(round(s*s,4))
    P = dataset.A.T/(1+sigma_2)
    P_val = str(round(P[0][0], 3))
    Q = sigma_2/ (1 + sigma_2)
    Q_val = str(round(Q,3))
    A_val = round(dataset.A[0][0], 3)
    M_val = str(round(vae.encoder.M.weight.detach().numpy()[0][0],3))
    W_val = str(round(vae.decoder.W.weight.detach().numpy()[0][0],3))
    S_val = str(round(torch.exp(vae.encoder.log_S).detach().numpy()[0][0],3))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.subplots_adjust(top=0.80)  # otherwise the right y-label is slightly clipped

    title ='''  {} Goal Values for Sample Complexity
                Case: D=d=1, N->Finite, s^2 - {}
                A = {} , sigma^2 = {} , s^2 = {} , P = {}, Q = {} '''.format(goal, Non, A_val, sigma_val, s_val, P_val, Q_val)
    file_name = '_'.join(title.split()) + '.png'
    plt.suptitle(title, color='red', y=0.98)
    plt.savefig(file_name)



    # plt.plot(epochs, vae_losses, label="VAE Objective")
    # # plt.plot(epochs, g_losses, label="E[g(W,s2)]")
    # # plt.plot(epochs, h_losses, label="E[h(M,S)]")
    # plt.legend()