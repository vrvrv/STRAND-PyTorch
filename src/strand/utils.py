from tqdm import tqdm
from typing import Sequence
from .functions import *
from sklearn.decomposition import NMF
from scipy.stats import poisson


class Initializer(object):
    def __init__(self,
                 rank: int,
                 k_dims: Sequence[int],
                 init: str = 'nmf',
                 nmf_max_iter: int = 1000,
                 **kwargs
                 ):

        self.rank = rank
        self.k_dims = k_dims

        self._init = init
        self.nmf_max_iter = nmf_max_iter

        self.buffers = dict()

    def init(self, count_tensor: torch.Tensor, feature=None):
        count_tensor = count_tensor.cpu()

        lamb = self.init_T0(count_tensor, method=self._init)

        if feature is not None:
            self.buffers['zeta'] = torch.eye(feature.size(0)).repeat((self.rank-1, 1, 1))
            self.buffers['sigma'] = torch.ones(self.rank - 1)
            self.buffers['Sigma_mat'] = torch.eye(self.rank - 1)

            self.buffers['Delta'] = self.buffers['Sigma_mat'].repeat((feature.size(1), 1, 1))
            self.buffers['lamb'] = lamb
            self.buffers['Xi'] = torch.normal(0, 1, size=(self.rank - 1, feature.size(0))).float()

        else:
            self.buffers['Lambda'] = logit_to_distribution(lamb)
            self.buffers['H'] = torch.ones(self.rank)

    def init_T0(self, count_tensor, method: str):
        V, D = count_tensor.size(-2), count_tensor.size(-1)

        nmf_trial = 15

        if method == 'nmf':
            y = count_tensor.sum(axis=(0, 1, 2, 3, 4))
            with tqdm(total=nmf_trial, desc=f'Initialization (NMF)', leave=True) as pbar:
                for i in range(nmf_trial):
                    nmf = NMF(
                        n_components=self.rank,
                        tol=1e-4,
                        init='random',
                        solver='mu',
                        beta_loss='kullback-leibler',
                        max_iter=self.nmf_max_iter
                    )

                    if i == 0:
                        T = nmf.fit_transform(y)
                        E = nmf.components_
                        ll = poisson((T @ E).flatten()).logpmf(y.flatten()).sum()

                    else:
                        Ti = nmf.fit_transform(y)
                        Ei = nmf.components_
                        lli = poisson((Ti @ Ei).flatten()).logpmf(y.flatten()).sum()
                        if lli > ll:
                            T, E = Ti, Ei
                            ll = lli

                    pbar.update(1)

                    pbar.set_postfix({'max_log_likelihood': ll})

            lamb = logit(E / np.where(E.sum(axis=0) < 1e-8, 1e-8, E.sum(axis=0)))
            _kt = torch.log(torch.from_numpy(T / T.sum(0)).float().clamp_(1e-10)).T
        elif method == 'random':
            lamb = torch.randn((self.rank - 1, D))
            _kt = torch.zeros(self.rank, V)
        else:
            raise NotImplementedError

        self.buffers['_kt'] = _kt

        self.buffers['_kc_t'] = torch.zeros((count_tensor.size(0), V))
        self.buffers['_kc_r'] = torch.zeros((count_tensor.size(1), V))
        self.buffers['_kc_e'] = torch.zeros((count_tensor.size(2), V))
        self.buffers['_kc_n'] = torch.zeros((count_tensor.size(3), V))
        self.buffers['_kc_c'] = torch.zeros((count_tensor.size(4), V))

        self.buffers['_ki_t'] = torch.zeros((count_tensor.size(0), V, self.rank))
        self.buffers['_ki_r'] = torch.zeros((count_tensor.size(1), V, self.rank))
        self.buffers['_ki_e'] = torch.zeros((count_tensor.size(2), V, self.rank))
        self.buffers['_ki_n'] = torch.zeros((count_tensor.size(3), V, self.rank))
        self.buffers['_ki_c'] = torch.zeros((count_tensor.size(4), V, self.rank))

        self.buffers['_kf_t'] = torch.zeros((count_tensor.size(0), self.rank))
        self.buffers['_kf_r'] = torch.zeros((count_tensor.size(1), self.rank))
        self.buffers['_kf_e'] = torch.zeros((count_tensor.size(2), self.rank))
        self.buffers['_kf_n'] = torch.zeros((count_tensor.size(3), self.rank))
        self.buffers['_kf_c'] = torch.zeros((count_tensor.size(4), self.rank))

        return lamb


def yphi_loader(Y, phi_d):
    yphi = Y[..., [0]] * phi_d(0) / Y.size(-1)
    for i in range(1, Y.size(-1)):
        yphi += Y[..., [i]] * phi_d(i) / Y.size(-1)
    return yphi
