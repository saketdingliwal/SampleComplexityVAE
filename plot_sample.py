import os,sys
import numpy as np 
import matplotlib.pyplot as plt

pre = '_2'
quantile = 0.85
goal = 'Adjusted'
ns = [50, 100, 250, 500, 1000, 5000, 10000, 50000]
vae_loss = np.load('vae{}.npy'.format(pre))
g_loss = np.load('g{}.npy'.format(pre))
# h_loss = np.load('h{}.npy'.format(pre))
# Ds = [1, 2, 3, 4, 5, 6, 7, 8]

# ns = vae_loss.shape[0]

vae_losses = []
g_losses = []
h_losses = []


# ns = [100, 500, 1000, 5000, 10000]

for idx,n in enumerate(ns):
    g = np.quantile(g_loss[idx], quantile)
    # h = np.quantile(h_loss[idx], quantile)
    vae = np.quantile(vae_loss[idx], quantile)

    vae_losses.append(vae)
    # g_losses.append(g)
    # g_losses.append(g/(d*np.sqrt(d)))
    g_losses.append(g*np.sqrt(n))
    # h_losses.append(h*n)
    # h_losses.append(h*n)


# ns = [50, 100, 500, 1000, 5000, 10000, 50000]





fig, ax1 = plt.subplots()
loss_color = 'green'
ax1.set_xlabel('number of samples (N)')
# ax1.set_xlabel('dimension (D=d)')
ax1.set_xscale('log')
ax1.set_ylabel('VAE loss', color=loss_color)
ax1.plot(ns, vae_losses, color=loss_color)
ax1.tick_params(axis='y', labelcolor=loss_color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

goal_color = 'blue'

# ax2.set_ylabel('g_hat(theta)/(d*sqrt(d))', color=goal_color)  # we already handled the x-label with ax1
ax2.set_ylabel('g_tilde * sqrt(N)', color=goal_color)  # we already handled the x-label with ax1
ax2.plot(ns, g_losses, color=goal_color)
ax2.tick_params(axis='y', labelcolor=goal_color)
title = '''
        Rate of Learning Analysis on N
        Case: D=d=1, delta = {}
        (Adjusted Goal * sqrt(N)) is decreasing
        '''.format(round(1-quantile,2))
# plt.title(title, color='red')
plt.suptitle(title, color='red')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.subplots_adjust(top=0.80)  # otherwise the right y-label is slightly clipped
plt.savefig('n_rate.png')
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

title = ''' Adjusted Goal Values for Sample Complexity Study
            Case: D=d=1, N->finite, s^2 - Non-trainable
            with probability at least {} the gold value is achieved for any n'''.format(quantile)

# title ='''  {} Goal Values for Sample Complexity
#             Case: D=d=1, N->Finite, s^2 - {}
#             A = {} , sigma^2 = {} , s^2 = {} , P = {}, Q = {} '''.format(goal, Non, A_val, sigma_val, s_val, P_val, Q_val)
# title = 'experiment 1 d=D=1 sample complexity'
file_name = '_'.join(title.split()) + '.png'
plt.suptitle(title, color='red', y=0.98)
plt.savefig('plot.png')