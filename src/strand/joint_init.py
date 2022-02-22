import math
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from tqdm import tqdm

from .functions import stack, factors_to_F, logit_to_distribution


def get_loss_fn(objective='poisson', tau=50):
    if objective == 'nbconst':
        def loss_fn(C, Chat):
            _Lij = tau \
                   * math.log(tau) \
                   - math.lgamma(tau) \
                   + torch.lgamma(C + tau) \
                   + C * torch.log(Chat) \
                   - torch.log(Chat + tau) \
                   * (tau + C) \
                   - torch.lgamma(C + 1)

            return - _Lij

    elif objective == 'poisson':
        def loss_fn(C, Chat):
            _Lij = C \
                   * torch.log(Chat) \
                   - Chat \
                   - torch.lgamma(C + 1)
            return - _Lij
    else:
        raise NotImplementedError

    return loss_fn


class Joint_INIT(nn.Module):
    def __init__(
            self,
            count_tensor,
            _T0,
            _t,
            _r,
            _e,
            _n,
            _c,
            _theta,
            **kwargs
    ):
        super(Joint_INIT, self).__init__()

        self.register_buffer('Y', count_tensor)

        self._T0 = nn.Parameter(_T0)
        self._theta = nn.Parameter(_theta)

        self._t = nn.Parameter(_t)
        self._r = nn.Parameter(_r)
        self._e = nn.Parameter(_e)
        self._n = nn.Parameter(_n)
        self._c = nn.Parameter(_c)

        self._at = nn.Parameter(torch.randn_like(_t))
        self._ar = nn.Parameter(torch.randn_like(_t))

        self.rank = _t.size(-1)

        self.loss_fn = get_loss_fn(objective='poisson')

    def forward(self, idx):

        Y_sub = self.Y[..., idx]

        T = stack(
            _T0=self._T0,
            _t=self._t,
            _r=self._r,
            e_dim=self._e.size(0) + 1,
            n_dim=self._n.size(0) + 1,
            c_dim=self._c.size(0) + 1
        )

        F = factors_to_F(
            _t=self._t, _r=self._r, _e=self._e, _n=self._n, _c=self._c, _at=self._at, _ar=self._ar,
            rank=self.rank, missing_rate=self.missing_rate, reduction=True
        )

        theta_sub = logit_to_distribution(self._theta[..., idx])

        chat = (T * F).matmul(theta_sub) * Y_sub.sum(dim=(0, 1, 2, 3, 4, 5))

        return self.loss_fn(Y_sub, chat).mean()

    def fit(self, batch_size, learning_rate, max_iter=1000, device='cpu'):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        indices = np.arange(self.Y.size(-1))

        pbar = tqdm(
            range(max_iter),
            desc='Initialization',
            leave=True,
            miniters=max_iter // 10
        )
        for _ in pbar:
            avg_loss = 0
            np.random.shuffle(indices)

            for i in range(math.ceil(self.Y.size(-1) / batch_size)):
                optimizer.zero_grad()
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(indices))
                random_idx = torch.from_numpy(indices[start_idx: end_idx]).long().to(device)

                loss = self(random_idx)
                loss.backward()

                optimizer.step()

                avg_loss += loss.item() * len(random_idx) / len(indices)

            pbar.set_postfix({'loss': avg_loss})

    @property
    def missing_rate(self):

        y_tr = self.Y.sum(dim=(2, 3, 4, -2, -1))

        _m00 = y_tr[:2, :2].sum(dim=(0, 1)).float() / y_tr.sum(dim=(0, 1))
        _m01 = y_tr[:2, 2].sum(dim=0).float() / y_tr.sum(dim=(0, 1))
        _m10 = y_tr[2, :2].sum(dim=0).float() / y_tr.sum(dim=(0, 1))
        _m11 = y_tr[2, 2].float() / y_tr.sum(dim=(0, 1))

        return torch.stack([_m00, _m01, _m10, _m11]).reshape(2, 2)
