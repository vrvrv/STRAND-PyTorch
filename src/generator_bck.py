import torch
import numpy as np

from scipy.stats import invgamma
from torch.distributions import *
from typing import Optional, Tuple, Union
from tqdm.auto import tqdm
from torch.nn import functional as F

from src.strand.functions import *


def isPD(A):
    E = np.linalg.eigvalsh(A)
    return np.all(E > 0)


class SimulationGenerator:
    def __init__(
            self,
            rank: int,
            dim: int,
            e_dim: int,
            n_dim: int,
            c_dim: int,
            mutation: int,
            total_sample: int,
            random_theta_generation: bool = True,
            feature_matrix_add_constant: bool = True,
            bt: Optional[Union[list, torch.Tensor]] = None,
            br: Optional[Union[list, torch.Tensor]] = None,
            epi: Optional[Union[list, torch.Tensor]] = None,
            nuc: Optional[Union[list, torch.Tensor]] = None,
            clu: Optional[Union[list, torch.Tensor]] = None,
            at: Optional[Union[list, torch.Tensor]] = None,
            ar: Optional[Union[list, torch.Tensor]] = None,
            T0: Optional[Union[list, torch.Tensor]] = None,
            m: Optional[Union[list, torch.Tensor]] = None,
            X: Optional[Union[list, torch.Tensor]] = None,
            Gamma: Optional[Union[list, torch.Tensor]] = None,
            nbinom: bool = False,
            disp_param: float = None,
            **kwargs
    ):

        self.rank = rank
        self.dim = dim
        self.e_dim = e_dim
        self.n_dim = n_dim
        self.c_dim = c_dim
        self.mutation = mutation
        self._total_sample = total_sample
        self.random_theta_generation = random_theta_generation
        self.feature_matrix_add_constant = feature_matrix_add_constant
        self.nbinom = nbinom
        self.disp_param = disp_param

        if isinstance(bt, list):
            bt = torch.Tensor(bt)
        self._bt = bt

        if isinstance(br, list):
            br = torch.Tensor(br)
        self._br = br

        if isinstance(at, list):
            at = torch.Tensor(at)
        self._at = at

        if isinstance(ar, list):
            ar = torch.Tensor(ar)
        self._ar = ar

        if isinstance(epi, list):
            epi = torch.Tensor(epi)
        self._epi = epi
        try:
            assert len(epi) == e_dim
        except AssertionError:
            print("given epi shape doesn't match")

        if isinstance(nuc, list):
            nuc = torch.Tensor(nuc)
        self._nuc = nuc
        try:
            assert len(nuc) == n_dim
        except AssertionError:
            print("given nuc shape doesn't match")

        if isinstance(clu, list):
            clu = torch.Tensor(clu)
        self._clu = clu
        try:
            assert len(clu) == c_dim
        except AssertionError:
            print("given clu shape doesn't match")

        if isinstance(m, list):
            m = torch.Tensor(m)
        self._m = m

        self._T0 = T0
        self._X = X
        self._Gamma = Gamma


        self.params = {
            'T0': self.T0,
            'factors': self.factors,
            'Gamma': self.Gamma
        }

        self._X = self.X

        self.params['theta'] = self.theta(
            mu=self.params['Gamma'].matmul(self._X)
        )

    @property
    def T0(self):
        if self._T0 is not None:
            return self._T0
        else:
            base_distribution = Dirichlet(0.3 * torch.ones((self.rank, self.mutation))).sample()
            CL, CG, TL, TG = Dirichlet(500 * base_distribution).sample((4,))
            T0 = torch.stack([CL.T, CG.T, TL.T, TG.T]).reshape(2, 2, self.mutation, self.rank)

        return T0

    @property
    def bt(self) -> torch.Tensor:
        if self._bt is not None:
            base_distribution = self._bt.repeat(self.rank, 1)
        else:
            base_distribution = Dirichlet(100 * torch.ones((self.rank, 2))).sample()

        return Dirichlet(100 * base_distribution).sample().T

    @property
    def br(self) -> torch.Tensor:
        if self._br is not None:
            base_distribution = self._br.repeat(self.rank, 1)
        else:
            base_distribution = Dirichlet(100 * torch.ones((self.rank, 2))).sample()

        return Dirichlet(100 * base_distribution).sample().T

    @property
    def at(self) -> torch.Tensor:
        if self._at is not None:
            base_distribution = self._at.repeat(self.rank, 1)
        else:
            base_distribution = Dirichlet(torch.ones((self.rank, 2))).sample()
            # base_distribution = Dirichlet(torch.ones((2, ))).sample()
        # return Dirichlet(15 * base_distribution).sample().T
        # return 0.5 * torch.ones((2, self.rank))
        return Dirichlet(10 * base_distribution).sample().T

    @property
    def ar(self) -> torch.Tensor:
        if self._ar is not None:
            base_distribution = self._ar.repeat(self.rank, 1)
        else:
            base_distribution = Dirichlet(torch.ones((self.rank, 2))).sample()
            # base_distribution = Dirichlet(torch.ones((2, ))).sample()
        # return Dirichlet(15 * base_distribution).sample().T
        # return 0.5 * torch.ones((2, self.rank))
        return Dirichlet(10 * base_distribution).sample().T

    @property
    def epi(self) -> torch.Tensor:
        if self._epi is not None:
            base_distribution = self._epi.repeat(self.rank, 1)
        else:
            base_distribution = Dirichlet(100 * torch.ones((self.rank, self.e_dim))).sample()

        return Dirichlet(30 * base_distribution).sample().T

    @property
    def nuc(self) -> torch.Tensor:
        if self._nuc is not None:
            base_distribution = self._nuc.repeat(self.rank, 1)
        else:
            base_distribution = Dirichlet(15 * torch.ones((self.rank, self.n_dim))).sample()

        return Dirichlet(15 * base_distribution).sample().T

    @property
    def clu(self) -> torch.Tensor:
        if self._clu is not None:
            base_distribution = self._clu.repeat(self.rank, 1)
        else:
            base_distribution = Dirichlet(15 * torch.ones((self.rank, self.c_dim))).sample()

        return Dirichlet(0.8 * base_distribution).sample().T

    @property
    def sigma(self):
        return invgamma(15).rvs(self.rank - 1)

    @property
    def Gamma(self):
        if self._Gamma is not None:
            return self._Gamma
        else:
            sigma = self.sigma
            Gamma = torch.empty((self.rank - 1, self.dim + 1))
            for k in range(self.rank - 1):
                Ip = torch.Tensor(np.eye(self.dim + 1))
                gamma_k = MultivariateNormal(
                    torch.zeros(self.dim + 1), sigma[k] * Ip
                ).sample()
                Gamma[k] = gamma_k

            return Gamma

    @property
    def X(self):
        if self._X is not None:
            X = self._X
        else:
            X = torch.randn((self.dim, self._total_sample))

        if self.feature_matrix_add_constant:
            X = torch.cat([X, torch.ones((1, X.size(dim=1)))], dim=0)
        return X

    def theta(self, mu):

        if self.random_theta_generation is True:

            A = Normal(0, 0.1).sample((self.rank - 1, self.rank - 1))
            Sigma = 0.2 * torch.eye(self.rank - 1) + (1 - torch.eye(self.rank - 1)) * A.matmul(A.T)

            it = 0
            while not isPD(Sigma):
                A = Normal(0, 0.1).sample((self.rank - 1, self.rank - 1))
                Sigma = 0.2 * torch.eye(self.rank - 1) + (1 - torch.eye(self.rank - 1)) * A.matmul(A.T)

                it += 1

                if it == 10:
                    raise NotImplementedError(f"{self.rank}")

            # A = Normal(0, 0.2).sample((self.rank - 1, self.rank - 1))
            # Sigma = torch.eye(self.rank - 1) + A.matmul(A.T)

            eta = MultivariateNormal(mu.T, Sigma).sample().T
            theta = logit_to_distribution(eta)

        else:
            tmp = torch.cat([mu, torch.zeros((1, mu.size(dim=1)))], dim=0)
            theta = F.softmax(tmp, dim=0)

        return theta

    @property
    def factors(self):
        return {
            't': self.bt,
            'r': self.br,
            'at': self.at,
            'ar': self.ar,
            'e': self.epi,
            'n': self.nuc,
            'c': self.clu
        }

    def _count_from_nb(self, n, mean):
        from scipy.stats import nbinom

        prob = n / (n + mean)
        count = torch.Tensor(nbinom(n, prob).rvs(self._total_sample))

        return count

    def sample(self, m):
        count = self._count_from_nb(
            n=50,
            mean=m
        )

        [cl, cg], [tl, tg] = self.params['T0']

        _T0 = torch.stack(
            [
                torch.stack([logit(cl), logit(cg)]),
                torch.stack([logit(tl), logit(tg)])
            ]
        )
        T = stack(
            _T0,
            logit(self.params['factors']['t']),
            logit(self.params['factors']['r']),
            self.e_dim,
            self.n_dim,
            self.c_dim
        )

        F = factors_to_F(
            logit(self.params['factors']['t']),
            logit(self.params['factors']['r']),
            logit(self.params['factors']['e']),
            logit(self.params['factors']['n']),
            logit(self.params['factors']['c']),
            logit(self.params['factors']['at']),
            logit(self.params['factors']['ar']),
            self.rank,
        )

        tf = T * F
        p_lv_k = tf.flatten(end_dim=-2)

        Y = torch.zeros((3, 3, self.e_dim, self.n_dim, self.c_dim, self.mutation, self._total_sample))

        for d, Yd in tqdm(enumerate(count), total=len(count), leave=False,
                          desc=f"rank : {self.rank}, n: {self._total_sample}, m: {m}"):
            if int(Yd) == 0:
                Yd += 1
            zd = Multinomial(total_count=int(Yd + 1), probs=self.params['theta'][:, d]).sample()

            for z, zcnt in enumerate(zd):
                if zcnt > 0:
                    if self.nbinom:
                        probs = p_lv_k[:, z] * zcnt / (p_lv_k[:, z] * zcnt + self.disp_param)

                        # Mean : p_lv_k[:, z] * zcnt
                        # Variance : Mean + Mean ** 2 / disp_param
                        v_cnt = NegativeBinomial(total_count=self.disp_param, probs=probs).sample()
                    else:
                        v_cnt = Multinomial(total_count=int(zcnt), probs=p_lv_k[:, z]).sample()
                    Y[..., d] += v_cnt.reshape(tf.shape[:-1])

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
            'feature': self._X.numpy(),
        }

        return data, self.params
