import math
import time
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from tqdm import tqdm


def safe_inverse(matrix: torch.Tensor, method='none', eps=1e-3):
    if method == 'none':
        inv = torch.inverse(matrix)
    elif method == 'spectral':
        e, Q = torch.eig(matrix, eigenvectors=True)
        P = e[:, 0]

        mask = P > eps

        Q = Q[:, mask]
        P = P[mask]

        inv = (Q.matmul(torch.diag(1 / P)).matmul(Q.T))
    else:
        raise NotImplementedError

    return inv.float()


def getDelta_d(lamb_d, Yn_d, SigmaInv):
    eta = torch.cat([lamb_d, torch.Tensor([0.])], dim=0)

    theta = torch.softmax(eta, dim=0)[:-1].reshape(-1, 1)

    neg_hessian = SigmaInv \
                  - Yn_d * theta.matmul(theta.T) \
                  + Yn_d * torch.diag(theta.squeeze(-1))

    Delta_d = torch.inverse(neg_hessian)

    return 0.001 * torch.diagonal(Delta_d) + 0.999 * Delta_d


def getDelta(lamb, Yphi, Sigma_inv):
    lamb = lamb.detach().cpu()
    Yphi = Yphi.detach().cpu()
    Sigma_inv = Sigma_inv.detach().cpu()

    K_1, D = lamb.shape

    Delta = torch.zeros((D, K_1, K_1))

    for d in tqdm(range(D), leave=False):
        Delta[d] = getDelta_d(
            lamb[:, d], Yphi[d].sum(), Sigma_inv
        )

    return Delta.float()


class LaplaceApproximation(nn.Module):
    def __init__(self, max_iter, batch_size):
        super(LaplaceApproximation, self).__init__()
        self.max_iter = max_iter
        self.batch_size = batch_size

    def forward(self, eta, mu, Yphi, Sigma_inv):

        loss1 = 0.5 * torch.diag((eta - mu).T.matmul(Sigma_inv).matmul(eta - mu))

        loss2 = - (Yphi * torch.log(
            torch.softmax(torch.cat([eta, torch.zeros(1, eta.size(1), device=eta.device)], dim=0), dim=0).T
        )).sum(-1)

        return loss1, loss2

    def fit(self, eta_init, mu, Yphi, Sigma, lr, inv_method='none', eps=1e-2, return_Delta=False, **kwargs):

        # Yphi.clamp_(min=1e-8)
        eta = nn.Parameter(eta_init)

        if not hasattr(self, 'optim'):
            # self.optim = optim.Adamax([eta], lr=lr, weight_decay=0.0001)
            self.optim = optim.RMSprop([eta], lr=lr)
        else:
            self.optim.param_groups[0]['params'] = [eta]

        batch_size = min(self.batch_size, mu.size(-1))
        indices = np.arange(mu.size(-1))

        pbar = tqdm(
            range(self.max_iter),
            desc='[E] Laplace Approximation',
            total=self.max_iter,
            leave=False,
            miniters=self.max_iter // 100
        )

        Sigma_inv = safe_inverse(Sigma, method=inv_method, eps=eps)

        for _ in pbar:
            avg_loss1 = 0
            avg_loss2 = 0
            for i in range(math.ceil(len(indices) / batch_size)):
                self.optim.zero_grad()
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(indices))

                train_idx = torch.from_numpy(indices[start_idx: end_idx]).long().to(mu.device)

                loss1, loss2 = self(eta[:, train_idx], mu[:, train_idx], Yphi[train_idx], Sigma_inv)

                loss = (loss1 + loss2).mean()
                loss.backward()

                self.optim.step()

                avg_loss1 += loss1.mean().item() * len(train_idx) / len(indices)
                avg_loss2 += loss2.mean().item() * len(train_idx) / len(indices)

            if _ % 100 == 0:
                pbar.set_postfix({'loss1': avg_loss1, 'loss2': avg_loss2})
                # print(eta[:, 1])

        lamb = eta.detach()
        if return_Delta:
            Delta = getDelta(lamb, Yphi, Sigma_inv).to(lamb.device)
        else:
            Delta = None
        return lamb, Delta
