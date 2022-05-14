import torch
import numpy as np
import h5py

from scipy.stats import invgamma
from torch.distributions import *
from typing import Optional, Tuple, Union
from tqdm.auto import tqdm
from torch.nn import functional as F

from src.strand.functions import *
from scipy.stats import nbinom, poisson


def isPD(A):
    E = np.linalg.eigvalsh(A)
    return np.all(E > 0)


class SimulationGenerator:
    def __init__(
            self,
            rank: int,
            context_dims: list,
            mutation: int,
            Gamma,
            total_sample: int,
            random_theta_generation: bool = True,
            nbinom: bool = False,
            disp_param: float = None,
            **kwargs
    ):

        self.p = 2
        self.rank = rank
        self.context_dims = context_dims
        self.mutation = mutation
        self._total_sample = total_sample
        self.random_theta_generation = random_theta_generation
        self.nbinom = nbinom
        self.disp_param = disp_param

        # with h5py.File("data/pcawg.hdf5", 'r') as f:
        #     snv = np.array(f['count_tensor']).sum((0, 1, 2, 3, 4, -1))
        #     snv = np.where(snv == 0, 1e-8, snv)

        # m = torch.log(torch.from_numpy(snv / snv.sum()).float())
        # m = m - m.mean()
        m = torch.zeros(96)

        _ks = torch.log(torch.distributions.Dirichlet(torch.ones(96) * 0.5).sample((self.rank, )))

        self.params = {
            '_m': m,
            '_ks': _ks,
            '_kc': self._kc,
            '_ki': self._ki,
            '_kf': self._kf,
            'X': self.X,
            'Gamma': Gamma
        }

        self.params['theta'] = self.theta(
            mu=self.params['Gamma'].matmul(self.params['X'])
        )

    @property
    def _kc(self):
        _kc = dict()
        for i, dim in enumerate(self.context_dims):
            _kc_i = torch.from_numpy(np.random.laplace(0, 0.5, size=(dim, 96))).float()
            # _kc_i = _kc_i * (torch.rand_like(_kc_i) > 0.8).float()

            _kc[f'_kc_{i}'] = _kc_i
        return _kc

    @property
    def _kf(self):
        _kf = dict()
        for i, dim in enumerate(self.context_dims):
            _kf[f'_kf_{i}'] = torch.from_numpy(np.random.laplace(0, 0.5, size=(dim, self.rank))).float()

        return _kf

    @property
    def _ki(self):
        _ki = dict()
        for i, dim in enumerate(self.context_dims):
            _ki[f'_ki_{i}'] = torch.zeros_like(torch.from_numpy(np.random.laplace(0, 0.1, size=(dim, self.rank))).float())
        return _ki

    # @property
    # def sigma(self):
    #     return invgamma(2).rvs(self.rank - 1)
    #
    # @property
    # def Gamma(self):
    #     sigma = self.sigma
    #     Gamma = torch.empty((self.rank - 1, self.p))
    #     for k in range(self.rank - 1):
    #         Ip = torch.Tensor(np.eye(self.p))
    #         gamma_k = MultivariateNormal(torch.zeros(self.p), sigma[k] * Ip).sample()
    #         Gamma[k] = gamma_k
    #
    #     return Gamma

    @property
    def X(self):
        x = np.random.uniform(size=self._total_sample)
        X = np.stack([
            x,
            np.sin(5 * x),
            np.cos(5 * x),
            np.log(0.01+x)
        ])
        return torch.from_numpy(X).float()

    def theta(self, mu):

        if self.random_theta_generation is True:

            # A = Normal(0, 0.1).sample((self.rank - 1, self.rank - 1))
            # Sigma = 0.2 * torch.eye(self.rank - 1) + (1 - torch.eye(self.rank - 1)) * A.matmul(A.T)
            #
            # it = 0
            # while not isPD(Sigma):
            #     A = Normal(0, 0.1).sample((self.rank - 1, self.rank - 1))
            #     Sigma = 0.2 * torch.eye(self.rank - 1) + (1 - torch.eye(self.rank - 1)) * A.matmul(A.T)
            #
            #     it += 1
            #
            #     if it == 10:
            #         raise NotImplementedError(f"{self.rank}")

            Sigma = 0.5 * torch.eye(self.rank - 1)

            # A = Normal(0, 0.2).sample((self.rank - 1, self.rank - 1))
            # Sigma = torch.eye(self.rank - 1) + A.matmul(A.T)

            # x = np.random.uniform(size=self._total_sample)
            # mu = np.stack([
            #     x,
            #     np.sin(5 * x),
            #     np.cos(5 * x),
            #     np.log(0.01+x)
            # ])
            #
            # mu = torch.from_numpy(mu).float()

            eta = MultivariateNormal(mu.T, Sigma).sample().T
            theta = logit_to_distribution(eta)

        else:
            tmp = torch.cat([mu, torch.zeros((1, mu.size(dim=1)))], dim=0)
            theta = F.softmax(tmp, dim=0)

        return theta

    def _count_from_nb(self, n, mean):

        # prob = n / (n + mean)
        # count = torch.Tensor(nbinom(n, prob).rvs(self._total_sample))
        count = torch.Tensor(poisson.rvs(mean, size=self._total_sample))
        return count

    def sample(self, m):
        count = self._count_from_nb(
            n=self.disp_param,
            mean=m
        )
        _kc = self.params['_kc']
        _kf = self.params['_kf']
        _ki = self.params['_ki']

        tf = topic_word_dist(
            _m=self.params['_m'],
            _kt=self.params['_ks'],
            _kc_t=_kc['_kc_0'],
            _kc_r=_kc['_kc_1'],
            _kc_e=_kc['_kc_2'],
            _kc_n=_kc['_kc_3'],
            _kc_c=_kc['_kc_4'],
            _kf_t=_kf['_kf_0'],
            _kf_r=_kf['_kf_1'],
            _kf_e=_kf['_kf_2'],
            _kf_n=_kf['_kf_3'],
            _kf_c=_kf['_kf_4'],
            _ki_t=_ki['_ki_0'],
            _ki_r=_ki['_ki_1'],
            _ki_e=_ki['_ki_2'],
            _ki_n=_ki['_ki_3'],
            _ki_c=_ki['_ki_4'],
        )

        p_lv_k = tf.flatten(end_dim=-2)
        p_lv_d = p_lv_k.matmul(self.params['theta'])

        Y = torch.zeros((*self.context_dims, self.mutation, self._total_sample))

        for d, Yd in tqdm(enumerate(count), total=len(count), leave=False,
                          desc=f"rank : {self.rank}, n: {self._total_sample}, m: {m}"):
            if int(Yd) == 0:
                Yd += 1
            Y[..., d] += Multinomial(total_count=int(Yd), probs=p_lv_d[:, d]).sample().reshape(tf.shape[:-1])

            # for z, zcnt in enumerate(zd):
            #     if zcnt > 0:
            #         if self.nbinom:
            #             probs = p_lv_k[:, z] * zcnt / (p_lv_k[:, z] * zcnt + self.disp_param)
            #
            #             # Mean : p_lv_k[:, z] * zcnt
            #             # Variance : Mean + Mean ** 2 / disp_param
            #             v_cnt = NegativeBinomial(total_count=self.disp_param, probs=probs).sample()
            #         else:
            #             v_cnt = Multinomial(total_count=int(zcnt), probs=p_lv_k[:, z]).sample()
            #         Y[..., d] += v_cnt.reshape(tf.shape[:-1])

        # theta = []
        # for d, Yd in tqdm(enumerate(count), total=len(count), leave=False,
        #                   desc=f"rank : {self.rank}, n: {self._total_sample}, m: {m}"):
        #     zd = torch.multinomial(self.params['theta'][:, d], int(Yd), replacement=True)
        #     theta_d = [int((zd == i).sum()) for i in range(self.rank)]
        #
        #     theta.append(theta_d)
        #
        # theta = torch.tensor(theta).T
        #
        # for k, Ck in enumerate(theta):
        #     p_lvk = p_lv_k[:, [k]]
        #     probs = p_lvk * Ck / (p_lvk * Ck + self.disp_param)
        #
        #     # Mean : p_lv_k[:, z] * zcnt
        #     # Variance : Mean + Mean ** 2 / disp_param
        #     v_cnt = NegativeBinomial(total_count=self.disp_param, probs=probs).sample()
        #
        #     Y += v_cnt.reshape(*tf.shape[:-1], -1)

        data = {
            'count_tensor': Y.numpy(),
            'feature': self.params['X'].numpy(),
        }

        return data, self.params, tf
