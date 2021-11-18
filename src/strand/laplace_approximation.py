import math
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from tqdm import tqdm


def getDelta_d(lamb_d, Yn_d, SigmaInv):
    eta = torch.cat([lamb_d, torch.Tensor([0.])], dim=0)

    theta = torch.softmax(eta, dim=0)[:-1].reshape(-1, 1)

    neg_hessian = SigmaInv \
                  - Yn_d * theta.matmul(theta.T) \
                  + Yn_d * torch.diag(theta.squeeze(-1))

    Delta_d = torch.inverse(neg_hessian)

    return 0.01 * torch.diagonal(Delta_d.clamp(0.01)) + 0.99 * Delta_d


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

        loss = 0.5 * torch.diag((eta - mu).T.matmul(Sigma_inv).matmul(eta - mu))

        loss = loss - (Yphi * torch.log(
            torch.softmax(
                torch.cat([eta, torch.zeros(1, eta.size(1), device=eta.device)], dim=0),
                dim=0
            ).T + 1e-10
        )).sum(-1)

        return loss.mean()

    def fit(self, eta_init, mu, Yphi, Sigma_inv, lr):

        Yphi.clamp_(min=0.001)
        eta = nn.Parameter(eta_init)

        if not hasattr(self, 'optim'):
            self.optim = optim.SGD([eta], lr=lr)
        else:
            self.optim.param_groups[0]['params'] = [eta]
            # self.optim.add_param_group({'params': [eta]})

        batch_size = min(self.batch_size, mu.size(-1))

        indices = np.arange(mu.size(-1))

        pbar = tqdm(
            range(self.max_iter),
            desc='[E] Laplace Approximation',
            total=self.max_iter,
            leave=False,
            miniters=10
        )

        for _ in pbar:
            avg_loss = 0
            np.random.shuffle(indices)
            for i in range(math.ceil(len(indices) / batch_size)):
                self.optim.zero_grad()
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, mu.size(-1))

                random_idx = torch.from_numpy(indices[start_idx: end_idx]).long().to(mu.device)

                loss = self(eta[:, random_idx], mu[:, random_idx], Yphi[random_idx], Sigma_inv)
                loss.backward()

                self.optim.step()

                avg_loss += loss.item() / (end_idx - start_idx + 1)

            if _ % 10 == 0:
                pbar.set_postfix({'avg_loss': avg_loss})

        lamb = eta.detach()
        Delta = getDelta(lamb, Yphi, Sigma_inv).to(lamb.device)

        return lamb, Delta
