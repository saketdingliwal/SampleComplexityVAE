import os,sys
import numpy as np 
import matplotlib.pyplot as plt


vae_loss = np.load('vae.npy')
g_loss = np.load('g.npy')
h_loss = np.load('h.npy')

ns = vae_loss.shape[0]

vae_losses = []
g_losses = []
h_losses = []

ns = [100, 500, 1000, 5000, 10000]

for idx,n in enumerate(ns):
    g = np.quantile(g_loss[idx], 0.95)
    h = np.quantile(h_loss[idx], 0.95)
    vae = np.quantile(vae_loss[idx], 0.95)

    vae_losses.append(vae)
    g_losses.append(g)
    h_losses.append(h)

goal = 'Gold'
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


# sigma_2 = dataset.sigma * dataset.sigma
# sigma_val = str(round(sigma_2,4))
# if s_trainable:
#     s_val = str(round(torch.exp(vae.decoder.log_s).detach().numpy()[0],3))
# else:
#     s_val = str(round(s*s,4))
# P = dataset.A.T/(1+sigma_2)
# P_val = str(round(P[0][0], 3))
# Q = sigma_2/ (1 + sigma_2)
# Q_val = str(round(Q,3))
# A_val = round(dataset.A[0][0], 3)
# M_val = str(round(vae.encoder.M.weight.detach().numpy()[0][0],3))
# W_val = str(round(vae.decoder.W.weight.detach().numpy()[0][0],3))
# S_val = str(round(torch.exp(vae.encoder.log_S).detach().numpy()[0][0],3))

fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.subplots_adjust(top=0.80)  # otherwise the right y-label is slightly clipped

# title ='''  {} Goal Values for Sample Complexity
#             Case: D=d=1, N->Finite, s^2 - {}
#             A = {} , sigma^2 = {} , s^2 = {} , P = {}, Q = {} '''.format(goal, Non, A_val, sigma_val, s_val, P_val, Q_val)
title = 'experiment 1 d=D=1 sample complexity'
file_name = '_'.join(title.split()) + '.png'
plt.suptitle(title, color='red', y=0.98)
plt.savefig(file_name)