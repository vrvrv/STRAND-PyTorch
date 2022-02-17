import os
import sys
import torch
import pickle
import matplotlib.pyplot as plt

from src.strand.model import STRAND
from src.strand.functions import *
from itertools import chain
from tqdm import tqdm

cs = torch.nn.CosineSimilarity(dim=0)

sid = 1
r_range = [5, 10, 20, 30]
n_range = [50, 100, 1000, 2000]
m_range = [50, 100, 1000, 2000]


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

    [[_cl, _cg], [_tl, _tg]] = pred['_T0']

    cl = logit_to_distribution(_cl)
    cg = logit_to_distribution(_cg)
    tl = logit_to_distribution(_tl)
    tg = logit_to_distribution(_tg)

    theta = logit_to_distribution(pred['lamb'])

    true_theta = param['theta']

    perm = match_signatures(true_theta.float(), theta.float())

    cl = torch.stack([cl[:, i] for i in perm], dim=1)
    cg = torch.stack([cg[:, i] for i in perm], dim=1)
    tl = torch.stack([tl[:, i] for i in perm], dim=1)
    tg = torch.stack([tg[:, i] for i in perm], dim=1)

    [[cl_true, cg_true], [tl_true, tg_true]] = param['T0']

    signature_recognition = list(
        chain(cs(cl_true, cl).tolist(),
              cs(cg_true, cg).tolist(),
              cs(tl_true, tl).tolist(),
              cs(tg_true, tg).tolist()
              )
    )
    return signature_recognition


def main():
    fig, ax = plt.subplots(4, 4, figsize=(16, 16), constrained_layout=True, sharey=True)
    fig.suptitle(
        "Accuracy of signature inference with respect to number of samples(n) and number of mutations per sample(m)",
        fontsize=16
    )

    with tqdm(total=len(n_range) * len(m_range) * len(r_range)) as pbar:

        for row, n in enumerate(n_range):
            for col, m in enumerate(m_range):

                sr_list = []

                for rank in r_range:
                    sr_list.append(
                        load_result(rank, n, m)
                    )
                    pbar.update(1)

                ax[row, col].boxplot(
                    sr_list, labels=r_range
                )

                ax[row, col].set_title("n : %d | m : %d" % (n, m))

                if row == 3:
                    ax[row, col].set_xlabel("Rank", fontsize=12)

                if col == 0:
                    ax[row, col].set_ylabel("Cosine Similarity", fontsize=12)

    plt.savefig("fig_a.pdf")


if __name__ == "__main__":
    main()
