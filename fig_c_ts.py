import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn import linear_model
from src.strand.functions import *

import argparse

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

    pred = torch.load(f'checkpoints/simulation_{sid}/rank_{rank}_n_{n}_m_{m}.ckpt', map_location=torch.device('cpu'))[
        'state_dict']

    theta_true = param['theta']
    theta_pred = logit_to_distribution(pred['lamb'])
    perm = match_signatures(theta_true, theta_pred.float())

    factors_pred = {
        't': torch.stack([logit_to_distribution(pred['_t'])[:, i] for i in perm], dim=-1),
        'r': torch.stack([logit_to_distribution(pred['_r'])[:, i] for i in perm], dim=-1),
        'e': torch.stack([logit_to_distribution(pred['_e'])[:, i] for i in perm], dim=-1),
        'n': torch.stack([logit_to_distribution(pred['_n'])[:, i] for i in perm], dim=-1),
        'c': torch.stack([logit_to_distribution(pred['_c'])[:, i] for i in perm], dim=-1)
    }

    factors_true = param['factors']

    return factors_true, factors_pred

def plot_c():
    fig, ax = plt.subplots(4, 4, figsize=(11, 11), constrained_layout=True, sharey=True)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    n_range = [50, 100, 1000, 2000]
    m_range = [50, 100, 1000, 2000]

    with tqdm(total=len(n_range) * len(m_range) * rank, desc="Draw Figure (C)") as pbar:
        for row, n in enumerate(n_range):
            for col, m in enumerate(m_range):
                x = []
                y = []

                true, pred = load_result(rank=rank, n=n, m=m)

                for i, f in enumerate(['t', 'r', 'e', 'n', 'c']):
                    if f == 't':
                        label = 'bt'
                    elif f == 'r':
                        label = 'br'
                    elif f == 'e':
                        label = 'k_epi'
                    elif f == 'n':
                        label = 'k_nuc'
                    else:
                        label = 'k_clu'

                    ax[row, col].scatter(
                        true[f][:-1], pred[f][:-1], s=50, alpha=0.3, c=colors[i], zorder=i + 1, label=label
                    )

                    x.append(true[f][:-1].flatten().numpy())
                    y.append(pred[f][:-1].flatten().numpy())

                    pbar.update(1)

                regr = linear_model.LinearRegression()
                x = np.hstack(x)
                y = np.hstack(y)
                regr.fit(x.reshape(-1, 1), y.reshape(-1, 1))
                pred = regr.predict(x.reshape(-1, 1))

                ax[row, col].plot(x, pred.squeeze(), color='r', linewidth=1.5, zorder=0)
                ax[row, col].set_title(
                    "n : %d | m : %d" % (n, m), fontsize=14
                )
                if row == len(n_range) - 1:
                    ax[row, col].set_xlabel("True factors", fontsize=13)

            ax[row, 0].set_ylabel("Inferred factors", fontsize=13)
        ax[row, col].legend(bbox_to_anchor=(1.4, 0.5), fontsize=8, title='Factors', title_fontsize=10)

    plt.savefig("fig_c.pdf")

if __name__ == "__main__":
    plot_c()