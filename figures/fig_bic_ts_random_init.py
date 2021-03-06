import os
import json
import math
import pickle
import argparse

from src.strand.functions import *
from tqdm import tqdm
from glob import glob
from torch.distributions import Poisson

# r_range = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
r_range = [17, 18]


def load_result(rank, Ysum, missing_rate, pred):
    k = 0

    _T0 = pred['_T0']
    _t = pred['_t']
    _r = pred['_r']
    _e = pred['_e']
    _n = pred['_n']
    _c = pred['_c']

    e_dim = _e.size(0) + 1
    n_dim = _n.size(0) + 1
    c_dim = _c.size(0) + 1

    T = stack(
        _T0=_T0,
        _t=_t,
        _r=_r,
        e_dim=e_dim,
        n_dim=n_dim,
        c_dim=c_dim,
        rank=rank
    ).float()

    k += torch.prod(torch.tensor(_T0.size())) \
         + torch.prod(torch.tensor(_t.size())) \
         + torch.prod(torch.tensor(_r.size())) \
         + torch.prod(torch.tensor(_e.size())) \
         + torch.prod(torch.tensor(_n.size())) \
         + torch.prod(torch.tensor(_c.size()))

    F = factors_to_F(
        _t=_t,
        _r=_r,
        _e=_e,
        _n=_n,
        _c=_c,
        e_dim=e_dim,
        n_dim=n_dim,
        c_dim=c_dim,
        missing_rate=missing_rate,
        rank=rank
    ).float()

    tf = (T * F).unsqueeze(-3)

    theta = logit_to_distribution(pred['lamb']).float()
    k += torch.prod(torch.tensor(pred['lamb'].size()))

    chat = (tf.matmul(theta) * Ysum.float()).squeeze().flatten(end_dim=-2)
    phat = tf.matmul(theta).squeeze().flatten(end_dim=-2)

    return phat.T, chat.T, k


def main(args):
    with open(f'data/ts/snv.pkl', 'rb') as f:
        Y = torch.from_numpy(pickle.load(f)).float()

    y = Y.flatten(end_dim=-2).T

    Ysum = Y.sum(dim=(0, 1, 2, 3, 4, 5))
    n = Y.size(-1)

    y_tr = Y.sum(dim=(2, 3, 4, -2, -1))

    _m00 = y_tr[:2, :2].sum(dim=(0, 1)).float() / y_tr.sum(dim=(0, 1))
    _m01 = y_tr[:2, 2].sum(dim=0).float() / y_tr.sum(dim=(0, 1))
    _m10 = y_tr[2, :2].sum(dim=0).float() / y_tr.sum(dim=(0, 1))
    _m11 = y_tr[2, 2].float() / y_tr.sum(dim=(0, 1))

    missing_rate = torch.stack([_m00, _m01, _m10, _m11]).reshape(2, 2)

    with tqdm(total=len(r_range), desc=f"Log BIC score") as pbar:
        bic_dict = dict()

        for rank in r_range:

            bic_dict[rank] = dict()

            for i, ckpt in enumerate(glob(f"checkpoints/ts/random_init/rank_{rank}_*")):

                pred = torch.load(ckpt, map_location=torch.device('cpu'))['state_dict']

                phat, chat, k = load_result(rank, Ysum=Ysum, missing_rate=missing_rate, pred=pred)

                ll = Poisson(chat.clamp(1e-8)).log_prob(y).sum()

                # ll = (y * torch.log(phat + 1e-20)).sum()

                # bic = - 2 * ll + k * math.log(n)

                bic_dict[rank][i] = {
                    'll': float(ll), 'k': int(k), 'n': int(n)
                }

                print(rank, i, ll)

            pbar.update(1)

    # with open(f"result/pcawg/bic_random_init.json", "w") as f:
    #     json.dump(bic_dict, f)
    #
    #     print("saved at result/pcawg/bic_random_init.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='draw figure b')
    parser.add_argument('--sid', type=int, default=1)
    args = parser.parse_args()
    main(args)
