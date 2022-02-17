import os
import json
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn import linear_model
from src.strand.functions import *
from torch.distributions import MultivariateNormal

cs = torch.nn.CosineSimilarity(dim=0)

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


def load_result(rank, n, m, sid):
    with open(f'data/simulation_{sid}/rank_{rank}_m_{m}_n_{n}.pkl', 'rb') as f:
        _, param = pickle.load(f)

    pred = torch.load(
        f'checkpoints/simulation_{sid}_ts/rank_{rank}_n_{n}_m_{m}_1.ckpt',
        map_location=torch.device('cpu')
    )['state_dict']

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

    theta_pred = abk.reshape(-1, 1) * E
    theta_pred = theta_pred / theta_pred.sum(0)
    theta_true = param['theta']

    perm = match_signatures(theta_true, theta_pred.float())

    theta_pred = torch.stack([theta_pred[i] for i in perm])

    return theta_true, theta_pred


def plot_b(args):
    fig, ax = plt.subplots(4, 4, figsize=(11, 11), constrained_layout=True, sharey=True)

    n_range = [50, 100, 1000, 2000]
    m_range = [50, 100, 1000, 2000]

    raw_data = dict()

    with tqdm(total=len(n_range) * len(m_range) * args.rank, desc=f"Fig B1 (Simulation {args.sid})") as pbar:
        for row, n in enumerate(n_range):
            for col, m in enumerate(m_range):
                x = []
                y = []

                true, pred = load_result(rank=args.rank, n=n, m=m, sid=args.sid)

                for r in range(args.rank):
                    ax[row, col].scatter(
                        true[r], pred[r], s=20, alpha=0.2, c='pink', zorder=0
                    )

                    x.append(true.numpy()[r])
                    y.append(pred.numpy()[r])

                    pbar.update(1)

                raw_data[f"n:{n},m:{m},rank:{args.rank}"] = {
                    'true': true.tolist(),
                    'pred': pred.tolist()
                }

                regr = linear_model.LinearRegression()
                x = np.hstack(x)
                y = np.hstack(y)
                regr.fit(x.reshape(-1, 1), y.reshape(-1, 1))
                pred = regr.predict(x.reshape(-1, 1))

                ax[row, col].plot(x, pred.squeeze(), color='r', linewidth=2, zorder=1)
                ax[row, col].set_title(
                    "n : %d | m : %d" % (n, m), fontsize=11  # 14
                )
                if row == len(n_range) - 1:
                    ax[row, col].set_xlabel("True exposure", fontsize=13)

            ax[row, 0].set_ylabel("Inferred exposure", fontsize=13)

    os.makedirs(f"assets/simulation_{args.sid}", exist_ok=True)
    output_file_nm = f"assets/simulation_{args.sid}/fig_b1_ts_rank_{args.rank}.pdf"
    output_file_nm_json = f"assets/simulation_{args.sid}/fig_b1_ts_rank_{args.rank}.json"

    plt.savefig(output_file_nm)

    with open(output_file_nm_json, "w") as f:
        json.dump(raw_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='draw figure b')
    parser.add_argument('--sid', type=int, default=1)
    parser.add_argument('--rank', type=int, default=5)
    args = parser.parse_args()

    plot_b(args)
