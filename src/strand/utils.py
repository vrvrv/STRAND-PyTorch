import torch
import numpy as np
from typing import Sequence
from .functions import *
from torch.distributions import Dirichlet
from sklearn.decomposition import NMF, non_negative_factorization

from scipy.linalg import solve_sylvester
from torch.utils.data import Dataset, DataLoader


class Initializer(object):
    def __init__(self,
                 rank: int,
                 k_dims: Sequence[int],
                 T0_init: str = 'nmf',
                 factors_init: str = 'nmf',
                 Xi_init: str = 'sylvester',
                 nmf_max_iter: int = 1000
                 ):

        self.rank = rank
        self.k_dims = k_dims

        self.T0_init = T0_init
        self.factors_init = factors_init
        self.Xi_init = Xi_init
        self.nmf_max_iter = nmf_max_iter

        self.buffers = dict()

    def fixed_nmf(self, X: np.array, W: np.array, H: np.array):
        w, h, _ = non_negative_factorization(
            X=X,
            n_components=self.rank,
            max_iter=self.nmf_max_iter,
            W=W,
            H=H,
            update_H=False,
            init='custom',
            solver='mu',
            beta_loss='kullback-leibler'
        )
        return w, h

    def init(self, count_tensor: torch.Tensor, feature=None, factor_names=None):
        count_tensor = count_tensor.cpu()

        theta = self.init_T0(count_tensor, method=self.T0_init)

        self.init_factors(count_tensor, theta, method=self.factors_init, factor_names=factor_names)

        _theta = logit(theta)

        if feature is not None:
            tmp = 0.1 * torch.rand(self.rank - 1, feature.size(0), 2)
            self.buffers['zeta'] = torch.eye(feature.size(0)) + tmp.matmul(tmp.transpose(-1, -2))
            self.buffers['sigma'] = torch.ones(self.rank - 1)

            tmp = 0.1 * torch.randn(self.rank - 1, self.rank - 1)
            self.buffers['Sigma_mat'] = torch.eye(self.rank - 1) + tmp.matmul(tmp.T)

            self.buffers['Delta'] = self.buffers['Sigma_mat'].repeat((feature.size(1), 1, 1))
            self.buffers['lamb'] = _theta

            if self.Xi_init == 'sylvester':
                A = (self.buffers['Sigma_mat'] / (self.buffers['sigma'] ** 2)).numpy()
                B = (feature.matmul(feature.T)).numpy()
                Q = (self.buffers['lamb'].float().matmul(feature.T)).numpy()

                Y = solve_sylvester(A, B, Q)

                self.buffers['Xi'] = torch.from_numpy(Y).float()

            elif self.Xi_init == 'random':
                self.buffers['Xi'] = torch.normal(0, 1, size=(self.rank - 1, feature.size(0))).double()

        else:
            self.buffers['Lambda'] = logit_to_distribution(_theta)
            self.buffers['H'] = torch.ones(self.rank)

    def init_T0(self, count_tensor, method: str):
        V, D = count_tensor.size(-2), count_tensor.size(-1)

        if method == 'nmf':
            nmf = NMF(
                n_components=self.rank,
                solver='mu',
                init='nndsvda',
                beta_loss='kullback-leibler',
                tol=1e-8,
                max_iter=self.nmf_max_iter
            )

            T = nmf.fit_transform(count_tensor.sum(axis=(0, 1, 2, 3, 4)).clamp_(0.01))
            E = nmf.components_

            theta = E / np.where(E.sum(axis=0) < 1e-8, 1e-8, E.sum(axis=0))

            # Fit T0_CL, T0_CG, T0_TL, T0_TG
            Y_cl = count_tensor[0, 0].sum(axis=(0, 1, 2)).numpy().astype(np.float64)
            Y_cg = count_tensor[0, 1].sum(axis=(0, 1, 2)).numpy().astype(np.float64)
            Y_tl = count_tensor[1, 0].sum(axis=(0, 1, 2)).numpy().astype(np.float64)
            Y_tg = count_tensor[1, 1].sum(axis=(0, 1, 2)).numpy().astype(np.float64)

            cl, _ = self.fixed_nmf(X=Y_cl, W=T, H=E)
            cg, _ = self.fixed_nmf(X=Y_cg, W=T, H=E)
            tl, _ = self.fixed_nmf(X=Y_tl, W=T, H=E)
            tg, _ = self.fixed_nmf(X=Y_tg, W=T, H=E)

            _cl = logit(torch.from_numpy(cl / cl.sum(axis=0)))
            _cg = logit(torch.from_numpy(cg / cg.sum(axis=0)))
            _tl = logit(torch.from_numpy(tl / tl.sum(axis=0)))
            _tg = logit(torch.from_numpy(tg / tg.sum(axis=0)))

        elif method == 'random':

            theta = Dirichlet(torch.ones((D, self.rank))).sample().T.numpy()

            _cl = logit(
                Dirichlet(torch.ones(self.rank, V)).sample().T
            )
            _cg = logit(
                Dirichlet(torch.ones(self.rank, V)).sample().T
            )
            _tl = logit(
                Dirichlet(torch.ones(self.rank, V)).sample().T
            )
            _tg = logit(
                Dirichlet(torch.ones(self.rank, V)).sample().T
            )
        else:
            raise NotImplementedError

        self.buffers['_T0'] = torch.stack([_cl, _cg, _tl, _tg]).reshape(2, 2, V - 1, self.rank)

        return theta

    def init_factors(self, count_tensor: torch.Tensor, theta: np.ndarray, method: str, factor_names):

        for i, (dim, fn) in enumerate(zip(self.k_dims, factor_names)):
            if method == 'nmf':
                Y_f = count_tensor.transpose(i, -2).sum(dim=(0, 1, 2, 3, 4))[:dim]
                Y_f = np.clip(Y_f, a_min=0.01, a_max=None)

                f, _, __ = non_negative_factorization(
                    n_components=self.rank,
                    X=Y_f / Y_f.sum(0),
                    W=None,
                    H=theta,
                    update_H=False,
                    solver='mu',
                    beta_loss='kullback-leibler'
                )

                _f = logit(torch.from_numpy(f / (f.sum(axis=0) + 0.0001)))

            elif method == 'random':

                _f = logit(
                    Dirichlet(torch.ones((self.rank, dim))).sample().T
                )

            else:
                raise NotImplementedError

            self.buffers[f"_{fn}"] = _f


def get_yphi_dataloader(Y, phi_d, batch_size):
    def collate_fn(batch):
        yphi = []
        idx = []
        for yphi_i, i in batch:
            yphi.append(yphi_i)
            idx.append(i)

        phi = torch.stack(yphi, dim=-3)
        idx = torch.tensor(idx, dtype=torch.long, device=phi.device)
        return phi, idx

    class yphi_iterator(Dataset):
        def __init__(self):
            super().__init__()
            self.Y = Y

        def __len__(self):
            return self.Y.size(-1)

        def __getitem__(self, item):
            return self.Y[..., [item]] * phi_d(item), item

    yphi = DataLoader(
        yphi_iterator(),
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    return yphi
