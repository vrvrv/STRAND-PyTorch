import os
import torch
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn import linear_model
from src.strand.functions import *
from torch.distributions import MultivariateNormal

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


def load_result_ts(rank, n, m):
    with open(f'data/simulation_{sid}/rank_{rank}_m_{m}_n_{n}.pkl', 'rb') as f:
        _, param = pickle.load(f)

    theta_true = param['theta']

    ssid = 1
    theta_pred = []
    while True:
        try:
            pred = torch.load(
                f'checkpoints/simulation_{sid}_ts/rank_{rank}_n_{n}_m_{m}_{ssid}.ckpt',
                map_location=torch.device('cpu')
            )['state_dict']

        except Exception as e:
            break
        E = torch.exp(pred['E0'])

        # A
        a1 = torch.exp(
            torch.cat([pred['a0'], pred['a0'], torch.zeros((2, rank))], dim=0)
        ).reshape(3, 2, rank)

        A = (a1[:, 0, :][:, None, :] * a1[:, 1, :][None, :, :]).reshape(
            3, 3, 1, 1, rank
        )

        # B
        B = torch.exp(
            torch.stack([
                pred['b0'][0, :] + pred['b0'][1, :], pred['b0'][0, :] - pred['b0'][1, :], pred['b0'][0, :],
                pred['b0'][1, :] - pred['b0'][0, :], pred['b0'][1, :] - pred['b0'][0, :], -pred['b0'][0, :],
                pred['b0'][1, :], -pred['b0'][1, :], torch.zeros(pred['b0'][0, :].shape)
            ]).reshape(3, 3, 1, 1, rank)
        )

        # K
        _cbiases = {}

        for i in range(3):
            _cbiases[i] = torch.cat(
                [torch.zeros([1, rank]), pred[f'k{i}']], dim=0
            )

        final_tensor = []

        card = np.array([12, 4, 2])
        card_prod = np.prod(card)

        arr_card_prod = np.arange(card_prod)

        C = card.flatten()
        idex = np.mod(
            np.floor(
                np.tile(arr_card_prod.flatten().T, (len(card), 1)).T /
                np.tile(np.cumprod(np.concatenate(([1.0], C[:0:-1]))),
                        (len(arr_card_prod), 1))),
            np.tile(C[::-1], (len(arr_card_prod), 1)))

        idex = idex[:, ::-1]

        for r in range(idex.shape[0]):
            current_term = []
            for c in range(idex.shape[1]):
                current_term.append(
                    _cbiases[c][idex[r, c].astype(int), :]
                )
            final_tensor.append(
                torch.stack(current_term).sum(dim=0)
            )

        K = torch.exp(
            torch.stack(final_tensor).reshape(1, 1, -1, 1, rank)
        )

        abk = (A * B * K).sum((0, 1, 2, 3))

        theta_i = abk.reshape(-1, 1) * E
        theta_i = theta_i / theta_i.sum(0)

        perm = match_signatures(theta_true, theta_i.float())

        theta_i = torch.stack([theta_i[j] for j in perm], dim=1)

        theta_pred.append(theta_i)

        ssid += 1

    theta_pred = torch.stack(theta_pred)

    _95_q = torch.quantile(theta_pred, q=0.975, dim=0).T
    _05_q = torch.quantile(theta_pred, q=0.025, dim=0).T

    return theta_true, (_05_q, _95_q)


def load_result_strand(rank, n, m):
    with open(f'data/simulation_{sid}/rank_{rank}_m_{m}_n_{n}.pkl', 'rb') as f:
        _, param = pickle.load(f)

    pred = torch.load(f'checkpoints/simulation_{sid}/rank_{rank}_n_{n}_m_{m}.ckpt', map_location=torch.device('cpu'))[
        'state_dict']

    pred_samples = MultivariateNormal(pred['lamb'].T.float(),
                                      pred['Delta'].mean(0).repeat(n, 1, 1).float()).rsample([1000, ])

    theta_pred_samples = torch.softmax(
        torch.cat([pred_samples, torch.zeros((1000, n, 1))], dim=-1),
        dim=-1
    )

    _95_q = torch.quantile(theta_pred_samples, q=0.975, dim=0).T
    _05_q = torch.quantile(theta_pred_samples, q=0.025, dim=0).T

    theta_true = param['theta']

    theta_pred = logit_to_distribution(pred['lamb'])
    perm = match_signatures(theta_true, theta_pred.float())

    theta_pred = torch.stack([theta_pred[i] for i in perm])

    _95_q = torch.stack([_95_q[i] for i in perm])
    _05_q = torch.stack([_05_q[i] for i in perm])

    return theta_true, (_05_q, _95_q)



def plot_b(args):

    fig, ax = plt.subplots(4, 4, figsize=(11, 11), constrained_layout=True, sharey=True)

    x_bin = [
        (i * 0.1, (i + 1) * 0.1) for i in range(10)
    ]

    n_range = [50, 100, 1000, 2000]
    m_range = [50, 100, 1000, 2000]

    with tqdm(total=len(n_range) * len(m_range), desc="Draw Figure (B)") as pbar:
        for row, n in enumerate(n_range):
            for col, m in enumerate(m_range):
                x = []
                y_ts = []
                y_strand = []

                true, (_05_q_ts, _95_q_ts) = load_result_ts(rank=rank, n=n, m=m)
                true, (_05_q_strand, _95_q_strand) = load_result_strand(rank=rank, n=n, m=m)

                for (l, u) in x_bin:
                    idx = torch.logical_and(l <= true, true < u)
                    coverage_ts = float(
                        torch.logical_and(_05_q_ts[idx] <= true[idx], true[idx] <= _95_q_ts[idx]).float().mean()
                    )
                    coverage_strand = float(
                        torch.logical_and(_05_q_strand[idx] <= true[idx], true[idx] <= _95_q_strand[idx]).float().mean()
                    )

                    x.append((l + u) / 2)
                    y_ts.append(coverage_ts)
                    y_strand.append(coverage_strand)

                pbar.update(1)

                ax[row, col].plot(x, y_ts, color='green', markersize=6, marker="+")
                ax[row, col].plot(x, y_strand, color='orange', markersize=6, marker="o")
                ax[row, col].set_title(
                    "n : %d | m : %d" % (n, m), fontsize=11  # 14
                )
                if row == len(n_range) - 1:
                    ax[row, col].set_xlabel("True exposure", fontsize=13)

            ax[row, 0].set_ylabel("Inferred exposure", fontsize=13)


    plt.savefig("fig_b_1_ts.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='draw figure b')
    parser.add_argument('--coverage', action='store_true')
    args = parser.parse_args()

    plot_b(args)
