import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn import linear_model
from src.strand.functions import *

cs = torch.nn.CosineSimilarity(dim=0)

sid = 1
rank = 5


def match_signatures(theta, theta_pred):
    """
    Args :
        theta : K x D
        theta_pred : K x D
    Returns : perm
        ex. perm = [0,2,3,1,4]
        theta_pred [0] <-> theta [0]
        theta_pred [2] <-> theta [1]
    """

    theta = theta.float()
    theta_pred = theta_pred.float()

    perm = []

    K = theta.size(0)

    for i in range(K):

        max_score = 0
        max_score_idx = 0

        for j in set(range(K)) - set(perm):
            score = theta[i].dot(theta_pred[j])
            score /= torch.sqrt((theta_pred[j] ** 2).sum())
            if score > max_score:
                max_score = score
                max_score_idx = j

        perm.append(max_score_idx)
    return perm


def load_result(rank, n, m):
    with open(f'data/simulation_{sid}/rank_{rank}_m_{m}_n_{n}.pkl', 'rb') as f:
        _, param = pickle.load(f)

    pred = torch.load(f'checkpoints/simulation_{sid}/rank_{rank}_n_{n}_m_{m}.ckpt', map_location=torch.device('cpu'))['state_dict']

    theta_true = param['theta']

    theta_pred = logit_to_distribution(pred['lamb'])
    perm = match_signatures(theta_true, theta_pred.float())

    theta_pred = torch.stack([theta_pred[i] for i in perm])
    return theta_true, theta_pred


fig, ax = plt.subplots(4, 4, figsize=(11, 11), constrained_layout=True, sharey=True)

n_range = [50, 100, 1000, 2000]
m_range = [50, 100, 1000, 2000]

with tqdm(total=len(n_range) * len(m_range) * rank, desc="Draw Figure (B)") as pbar:
    for row, n in enumerate(n_range):
        for col, m in enumerate(m_range):
            x = []
            y = []

            true, pred = load_result(rank=rank, n=n, m=m)

            for r in range(rank):
                ax[row, col].scatter(
                    true[r], pred[r], s=20, alpha=0.2, c='pink', zorder=0
                )

                x.append(true.numpy()[r])
                y.append(pred.numpy()[r])

                pbar.update(1)

            regr = linear_model.LinearRegression()
            x = np.hstack(x)
            y = np.hstack(y)
            regr.fit(x.reshape(-1, 1), y.reshape(-1, 1))
            pred = regr.predict(x.reshape(-1, 1))

            ax[row, col].plot(x, pred.squeeze(), color='r', linewidth=2, zorder=1)
            ax[row, col].set_title(
                "n : %d | m : %d" % (n, m), fontsize=14
            )
            if row == len(n_range) - 1:
                ax[row, col].set_xlabel("True exposure", fontsize=13)

        ax[row, 0].set_ylabel("Inferred exposure", fontsize=13)
        
plt.savefig("fig_b.pdf")
