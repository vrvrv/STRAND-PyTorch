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


def load_result(rank, n, m, sid):
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
    return theta_true, theta_pred, (_05_q, _95_q)


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

                true, pred, (_05_q, _95_q) = load_result(rank=args.rank, n=n, m=m, sid=args.sid)

                for r in range(args.rank):
                    ax[row, col].scatter(
                        true[r], pred[r], s=20, alpha=0.2, c='pink', zorder=0
                    )

                    x.append(true.numpy()[r])
                    y.append(pred.numpy()[r])

                    pbar.update(1)

                coverage = float(torch.logical_and(_05_q <= true, true <= _95_q).float().mean())

                raw_data[f"n:{n},m:{m},rank:{args.rank}"] = {
                    'true': true.tolist(),
                    'pred': pred.tolist(),
                    '0.05q': _05_q.tolist(),
                    '0.95q': _95_q.tolist(),
                    'coverage': coverage
                }

                regr = linear_model.LinearRegression()
                x = np.hstack(x)
                y = np.hstack(y)
                regr.fit(x.reshape(-1, 1), y.reshape(-1, 1))
                pred = regr.predict(x.reshape(-1, 1))

                ax[row, col].plot(x, pred.squeeze(), color='r', linewidth=2, zorder=1)

                ax[row, col].set_title(
                    "n : %d | m : %d | coverage : %.2f" % (n, m, coverage), fontsize=11  # 14
                )
                if row == len(n_range) - 1:
                    ax[row, col].set_xlabel("True exposure", fontsize=13)

            ax[row, 0].set_ylabel("Inferred exposure", fontsize=13)

    os.makedirs(f"assets/simulation_{args.sid}", exist_ok=True)

    output_file_nm = f"assets/simulation_{args.sid}/fig_b1_rank_{args.rank}.pdf"
    output_file_nm_json = f"assets/simulation_{args.sid}/fig_b1_rank_{args.rank}.json"

    plt.savefig(output_file_nm)

    with open(output_file_nm_json, "w") as f:
        json.dump(raw_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='draw figure b')
    parser.add_argument('--sid', type=int, default=1)
    parser.add_argument('--rank', type=int, default=5)
    parser.add_argument('--coverage', action='store_true')
    args = parser.parse_args()

    plot_b(args)
