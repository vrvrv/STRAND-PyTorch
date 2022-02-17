import pickle
import logging
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Dirichlet

from .functions import *
from .laplace_approximation import LaplaceApproximation
from .tnf import TF_opt_model

from scipy.linalg import solve_sylvester
from sklearn.decomposition import NMF, non_negative_factorization

from tqdm import tqdm
from typing import Tuple

logger = logging.getLogger('src.train')


def collate_fn(batch):
    yphi = []
    idx = []
    for yphi_i, i in batch:
        yphi.append(yphi_i)
        idx.append(i)

    phi = torch.stack(yphi, dim=-3)
    idx = torch.tensor(idx, dtype=torch.long, device=phi.device)
    return phi, idx


def deviance(pred, true):
    d_ij = 2 * (pred * torch.log(pred / (true + 1e-8) + 1e-8) - pred + true)
    return d_ij.mean()


class STRAND(pl.LightningModule):
    def __init__(self,
                 data_dir: dict,
                 rank: int,
                 t_dim: int,
                 r_dim: int,
                 e_dim: int,
                 n_dim: int,
                 c_dim: int,
                 laplace_approx_conf: dict,
                 use_covariate: bool = True,
                 init: str = 'NMF',
                 nmf_max_iter: int = 10000,
                 e_iter: int = 20,
                 tf_lr: float = 0.1,
                 tf_batch_size: int = 32,
                 tf_max_steps: int = 200,
                 bias_correction: bool = True,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.use_covariate:
            with open(data_dir['X'], "rb") as f:
                self.register_buffer(
                    'X', torch.from_numpy(pickle.load(f)).float(), persistent=False
                )

                self.hparams['p'] = self.X.shape[0]

        with open(data_dir['Y'], "rb") as f:
            self.register_buffer(
                'Y', torch.from_numpy(pickle.load(f)).float(), persistent=False
            )
            self.hparams['V'], self.hparams['D'] = self.Y.size(-2), self.Y.size(-1)

        if self.hparams.init == 'NMF':
            self.nmf_init()
        elif self.hparams.init == 'Random':
            self.random_init()
        else:
            raise NotImplementedError

        self.la = LaplaceApproximation(
            max_iter=self.hparams.laplace_approx_conf.max_iter,
            batch_size=self.hparams.laplace_approx_conf.batch_size
        )

        self.tnf = TF_opt_model(
            Y=self.Y, _T0=self._T0,
            _t=self._t, _r=self._r, _e=self._e, _n=self._n, _c=self._c,
            e_dim=self.hparams.e_dim, n_dim=self.hparams.n_dim, c_dim=self.hparams.c_dim
        )

        self.automatic_optimization = False

    def nmf_init(self):

        def fixed_NMF(X: np.array, W: np.array, H: np.array):
            w, h, _ = non_negative_factorization(
                X=X,
                n_components=self.hparams.rank,
                max_iter=self.hparams.nmf_max_iter,
                W=W,
                H=H,
                update_H=False,
                init='custom',
                solver='mu',
                beta_loss='kullback-leibler'
            )
            return w, h

        nmf = NMF(
            n_components=self.hparams.rank,
            solver='mu',
            init='nndsvda',
            beta_loss='kullback-leibler',
            tol=1e-8,
            max_iter=self.hparams.nmf_max_iter
        )

        T = nmf.fit_transform(self.Y.sum(axis=(0, 1, 2, 3, 4)).clamp_(0.01))
        E = nmf.components_

        # T, E = fixed_NMF(X=self.Y.sum(axis=(0, 1, 2, 3, 4)), W=T, H=E)

        theta = torch.from_numpy(E / np.where(E.sum(axis=0) < 1e-8, 1e-8, E.sum(axis=0)))

        # Fit T0_CL, T0_CG, T0_TL, T0_TG
        Y_cl = self.Y[0, 0].cpu().sum(axis=(0, 1, 2)).numpy().astype(np.float64)
        Y_cl_sum = Y_cl.sum(0)
        Y_cl_idx = np.arange(self.hparams.D)[Y_cl_sum > np.quantile(Y_cl_sum, 0.1)]

        Y_cg = self.Y[0, 1].cpu().sum(axis=(0, 1, 2)).numpy().astype(np.float64)
        Y_cg_sum = Y_cg.sum(0)
        Y_cg_idx = np.arange(self.hparams.D)[Y_cg_sum > np.quantile(Y_cg_sum, 0.1)]

        Y_tl = self.Y[1, 0].cpu().sum(axis=(0, 1, 2)).numpy().astype(np.float64)
        Y_tl_sum = Y_tl.sum(0)
        Y_tl_idx = np.arange(self.hparams.D)[Y_tl_sum > np.quantile(Y_tl_sum, 0.1)]

        Y_tg = self.Y[1, 1].cpu().sum(axis=(0, 1, 2)).numpy().astype(np.float64)
        Y_tg_sum = Y_tg.sum(0)
        Y_tg_idx = np.arange(self.hparams.D)[Y_tg_sum > np.quantile(Y_tg_sum, 0.1)]

        cl, _ = fixed_NMF(X=Y_cl[:, Y_cl_idx], W=T, H=E[:, Y_cl_idx])
        cg, _ = fixed_NMF(X=Y_cg[:, Y_cg_idx], W=T, H=E[:, Y_cg_idx])
        tl, _ = fixed_NMF(X=Y_tl[:, Y_tl_idx], W=T, H=E[:, Y_tl_idx])
        tg, _ = fixed_NMF(X=Y_tg[:, Y_tg_idx], W=T, H=E[:, Y_tg_idx])

        _cl = logit(torch.from_numpy(cl / cl.sum(axis=0)))
        _cg = logit(torch.from_numpy(cg / cg.sum(axis=0)))
        _tl = logit(torch.from_numpy(tl / tl.sum(axis=0)))
        _tg = logit(torch.from_numpy(tg / tg.sum(axis=0)))

        self.register_buffer(
            '_T0', torch.stack([_cl, _cg, _tl, _tg]).reshape(2, 2, self.hparams.V - 1, self.hparams.rank)
        )

        _theta = logit(theta)

        logger.info("Initialized T0")

        for i, factor in enumerate(['t', 'r', 'e', 'n', 'c']):
            logger.info(f"Intializing factor, {factor}")

            Y_f = self.Y.transpose(i, -2).sum(dim=(0, 1, 2, 3, 4))

            if factor in {'t', 'r'}:
                # Y_f[0] = Y_f[0] / (self.missing_rate[0, 0] + self.missing_rate[0, 1])
                # Y_f[1] = Y_f[1] / (self.missing_rate[0, 0] + self.missing_rate[1, 0])

                Y_f = Y_f[:2]

            f, _, __ = non_negative_factorization(
                n_components=self.hparams.rank,
                X=Y_f.clamp(0.1) / Y_f.clamp(0.1).sum(0),
                W=None,
                H=E,
                solver='mu',
                beta_loss='kullback-leibler',
                update_H=False
            )

            self.register_buffer(
                f'_{factor}', logit(torch.from_numpy(f / (f.sum(axis=0) + 0.0001)))
            )

        logger.info("Initialized factors t, r, e, n, c")

        if self.hparams.use_covariate:
            logger.info("Initialize zeta")
            tmp = 0.1 * torch.rand(self.hparams.rank - 1, self.hparams.p, 2)
            self.register_buffer(
                'zeta', torch.eye(self.hparams.p) + tmp.matmul(tmp.transpose(-1, -2))
            )

            logger.info("Initialize sigma")
            self.register_buffer(
                'sigma', torch.ones(self.hparams.rank - 1)
            )

            logger.info("Intialize Sigma_mat")
            tmp = torch.randn(self.hparams.rank - 1, self.hparams.rank - 1)
            self.register_buffer(
                'Sigma_mat', torch.eye(self.hparams.rank - 1) + tmp.matmul(tmp.T)
            )

            logger.info("Initialize Delta")
            self.register_buffer(
                'Delta', self.Sigma_mat.repeat((self.hparams.D, 1, 1))
            )

            logger.info("Initialize lambda")
            self.register_buffer(
                'lamb', _theta
            )

            logger.info("Initialize Xi")
            A = (self.Sigma_mat / (self.sigma ** 2)).numpy()
            B = (self.X.matmul(self.X.T)).numpy()
            Q = (self.lamb.float().matmul(self.X.T)).numpy()

            Y = solve_sylvester(A, B, Q)

            self.register_buffer(
                'Xi', torch.from_numpy(Y).float()
            )

        else:
            logger.info("Initialize Lambda")
            self.register_buffer(
                'Lambda', logit_to_distribution(_theta)
            )

            logger.info("Initialize H")
            self.register_buffer(
                'H', torch.ones(self.hparams.rank)
            )

        logger.info("Initialization Ended")

    def random_init(self) -> None:

        _cl = logit(
            Dirichlet(torch.ones(self.hparams.rank, self.hparams.V)).sample().T
        )
        _cg = logit(
            Dirichlet(torch.ones(self.hparams.rank, self.hparams.V)).sample().T
        )
        _tl = logit(
            Dirichlet(torch.ones(self.hparams.rank, self.hparams.V)).sample().T
        )
        _tg = logit(
            Dirichlet(torch.ones(self.hparams.rank, self.hparams.V)).sample().T
        )

        self.register_buffer(
            '_T0', torch.stack([_cl, _cg, _tl, _tg]).reshape(2, 2, self.hparams.V - 1, self.hparams.rank)
        )

        _theta = logit(
            Dirichlet(torch.ones((self.hparams.D, self.hparams.rank))).sample().T
        )

        for i, factor in enumerate(['t', 'r', 'e', 'n', 'c']):
            dim = getattr(self.hparams, f"{factor}_dim")

            _f = logit(
                Dirichlet(torch.ones((self.hparams.rank, dim))).sample().T
            )

            self.register_buffer(
                f'_{factor}', _f
            )

        if self.X is not None:
            tmp = 0.1 * torch.rand(self.hparams.rank - 1, self.hparams.p, 2)
            self.register_buffer(
                'zeta', torch.eye(self.hparams.p) + tmp.matmul(tmp.transpose(-1, -2))
            )

            self.register_buffer(
                'sigma', torch.ones(self.hparams.rank - 1)
            )

            tmp = torch.randn(self.hparams.rank - 1, self.hparams.rank - 1)
            self.register_buffer(
                'Sigma_mat', (torch.eye(self.hparams.rank - 1) + tmp.matmul(tmp.T)).float()
            )

            self.register_buffer(
                'Delta', self.Sigma_mat.repeat((self.hparams.D, 1, 1))
            )

            self.register_buffer(
                'lamb', _theta
            )

            self.register_buffer(
                'Xi', torch.normal(0, 1, size=(self.hparams.rank - 1, self.hparams.p)).double()
            )

        else:
            self.register_buffer(
                'Lambda', logit_to_distribution(_theta)
            )

            self.register_buffer(
                'H', torch.ones(self.hparams.rank)
            )

    def e_step(self):

        if self.hparams.use_covariate:
            # UPDATE zeta
            for k in range(self.hparams.rank - 1):
                self.zeta[k] = torch.inverse(
                    1 / (self.sigma[k] ** 2) * torch.eye(self.hparams.p, device=self.device) \
                    + self.Sigma_inv[k, k] * self.X.matmul(self.X.T)
                ).float()

            # Update Xi, eta and Delta
            pbar = tqdm(range(self.hparams.e_iter), desc=f'E-STEP {self.current_epoch}', leave=False)
            for _ in pbar:
                # Update Xi
                A = (self.Sigma_mat / (self.sigma ** 2)).cpu().numpy()
                B = (self.X.matmul(self.X.T)).cpu().numpy()
                Q = (self.lamb.float().matmul(self.X.T)).cpu().numpy()

                self.Xi = torch.from_numpy(
                    solve_sylvester(A, B, Q)
                ).float().to(self.device)

                # Update eta and Delta
                self.lamb, self.Delta = self.la.fit(
                    eta_init=self.lamb,
                    mu=self.mu,
                    Yphi=self.Yphi,
                    Sigma=self.Sigma_mat,
                    lr=self.hparams.laplace_approx_conf.lr,
                    inv_method=self.hparams.laplace_approx_conf.inv_method,
                    eps=self.hparams.laplace_approx_conf.eps
                )
                # pbar.set_postfix({'negative_elbo': self.negative_elbo})

        else:
            # Update Lambda
            self.Lambda = (self.H + self.Yphi).T

    def m_step(self):
        if self.hparams.use_covariate:
            Sigma_mat = self.Delta.mean(dim=0)

            disc = (self.lamb - self.mu).T.unsqueeze(-1)
            tmp = disc.matmul(disc.transpose(-1, -2))

            Sigma_mat += tmp.mean(axis=0)

            tmp = self.X.T.matmul(self.zeta.float()).matmul(self.X)
            tmp = torch.diagonal(tmp, dim1=1, dim2=2).mean(dim=1)
            tmp = torch.diag_embed(tmp.float())

            Sigma_mat += tmp

            self.Sigma_mat = 0.001 * torch.diagonal(Sigma_mat) + 0.999 * Sigma_mat

            # Update sigma
            for k in range(self.hparams.rank - 1):
                sigma_k = torch.sqrt((torch.trace(self.zeta[k]) + (self.Xi[k] ** 2).sum()) / self.hparams.p)
                self.sigma[k] = sigma_k.clamp_(0.001)

        # else:
        #     self.H = self.Lambda.sum(1) / self.Lambda.sum()

        with torch.no_grad():
            self.tnf._T0 = nn.Parameter(self._T0)
            self.tnf._t = nn.Parameter(self._t)
            self.tnf._r = nn.Parameter(self._r)
            self.tnf._e = nn.Parameter(self._e)
            self.tnf._n = nn.Parameter(self._n)
            self.tnf._c = nn.Parameter(self._c)

        class yphi_iterator(Dataset):
            def __init__(self_phi):
                super(yphi_iterator, self_phi).__init__()

            def __len__(self_phi):
                return self.hparams.D

            def __getitem__(self_phi, item):
                return self.Y[..., [item]] * self.phi_d(item), item

        yphi = DataLoader(
            yphi_iterator(),
            batch_size=self.hparams.tf_batch_size,
            collate_fn=collate_fn
        )

        self.tnf.fit(
            yphi_loader=yphi,
            lr=self.hparams.tf_lr,
            max_steps=self.hparams.tf_max_steps
        )

        self._T0 = 0.99 * self.tnf._T0 + 0.01 * self._T0

        for k in ['t', 'r', 'e', 'n', 'c']:
            setattr(self, f"_{k}", getattr(self.tnf, f"_{k}"))

    def training_step(self, *args, **kwargs):
        self.e_step()
        self.m_step()

        self.log("negative_elbo", self.negative_elbo, on_epoch=True, prog_bar=True, logger=True)

    def validataion_step(self, *args, **kwargs):
        Y = self.Y.reshape(-1, 96)
        non_zero_idx = Y.sum(dim=-1) != 0
        Y = Y[non_zero_idx]

        Chat_flatten = self.Chat.reshape(-1, 96)

        p = Y / Y.sum(dim=-1, keepdim=True)
        phat = Chat_flatten[non_zero_idx] / Chat_flatten[non_zero_idx].sum(dim=-1, keepdim=True)

        p_error_1 = torch.abs(p - phat).sum(dim=-1)
        p_error_2 = torch.sqrt(torch.square(p - phat).sum(dim=-1))

        self.log_dict(
            {
                "p_error_1_avg": p_error_1.mean(),
                "p_error_1_std": p_error_1.std(),
                "p_error_2_avg": p_error_2.mean(),
                "p_error_2_std": p_error_2.std(),
                "cross_entropy": - (p * torch.log(phat)).mean()
            },
            logger=True
        )

    def test_step(self, *args, **kwargs):
        dev = deviance(self.Chat, self.Y)

        Y = self.Y.reshape(-1, 96)
        non_zero_idx = Y.sum(dim=-1) != 0
        Y = Y[non_zero_idx]

        Chat_flatten = self.Chat.reshape(-1, 96)

        p = Y / Y.sum(dim=-1, keepdim=True)
        phat = Chat_flatten[non_zero_idx] / Chat_flatten[non_zero_idx].sum(dim=-1, keepdim=True)

        p_error_1 = torch.abs(p - phat).sum(dim=-1)
        p_error_2 = torch.sqrt(torch.square(p - phat).sum(dim=-1))

        self.log_dict(
            {
                "deviance": dev,
                "p_error_1_avg": p_error_1.mean(),
                "p_error_1_std": p_error_1.std(),
                "p_error_2_avg": p_error_2.mean(),
                "p_error_2_std": p_error_2.std(),
                "cross_entropy": - (p * torch.log(phat)).mean()
            },
            logger=True
        )

    @property
    def Sigma_inv(self):
        return torch.inverse(self.Sigma_mat).float()

    @property
    def mu(self):
        return self.Xi.matmul(self.X).float()

    @property
    def T(self):
        return stack(
            _T0=self._T0,
            _t=self._t,
            _r=self._r,
            e_dim=self.hparams.e_dim,
            n_dim=self.hparams.n_dim,
            c_dim=self.hparams.c_dim,
            rank=self.hparams.rank
        )

    @property
    def F(self):
        return factors_to_F(
            _t=self._t,
            _r=self._r,
            _e=self._e,
            _n=self._n,
            _c=self._c,
            e_dim=self.hparams.e_dim,
            n_dim=self.hparams.n_dim,
            c_dim=self.hparams.c_dim,
            missing_rate=self.missing_rate,
            rank=self.hparams.rank
        )

    @property
    def phi(self) -> torch.Tensor:
        if self.hparams.use_covariate:
            phi = Phi(
                T=self.T, F=self.F, lambda_or_Lambda=('lambda', self.lamb)
            )
        else:
            phi = Phi(
                T=self.T, F=self.F, lambda_or_Lambda=('Lambda', self.Lambda)
            )

        return phi.float()

    def phi_d(self, d):
        if self.hparams.use_covariate:
            phi_d = Phi_d(
                T=self.T, F=self.F, lambda_or_Lambda=('lambda', self.lamb[:, d])
            )

        else:
            phi_d = Phi_d(
                T=self.T, F=self.F, lambda_or_Lambda=('Lambda', self.Lambda[:, d])
            )

        return phi_d

    def F_d(self, d):
        y_tr = self.Y[..., d].sum(dim=(2, 3, 4, -1))

        _m00_d = y_tr[:2, :2].sum(dim=(0, 1)).float() / y_tr.sum(dim=(0, 1))
        _m01_d = y_tr[:2, 2].sum(dim=0).float() / y_tr.sum(dim=(0, 1))
        _m10_d = y_tr[2, :2].sum(dim=0).float() / y_tr.sum(dim=(0, 1))
        _m11_d = y_tr[2, 2].float() / y_tr.sum(dim=(0, 1))

        m_d = torch.stack([_m00_d, _m01_d, _m10_d, _m11_d]).reshape(2, 2)

        F_d = factors_to_F(
            _t=self._t,
            _r=self._r,
            _e=self._e,
            _n=self._n,
            _c=self._c,
            e_dim=self.hparams.e_dim,
            n_dim=self.hparams.n_dim,
            c_dim=self.hparams.c_dim,
            missing_rate=m_d,
            rank=self.hparams.rank,
            reduction=True
        )

        return F_d

    @property
    def Yphi(self):

        if self.hparams.use_covariate:
            yphi = Yphi(
                Y=self.Y, T=self.T, F=self.F, lambda_or_Lambda=('lambda', self.lamb)
            )
        else:
            yphi = Yphi(
                Y=self.Y, T=self.T, F=self.F, lambda_or_Lambda=('Lambda', self.Lambda)
            )

        return yphi

    @property
    def tf(self):
        _tf = self.T.cpu() * self.F.cpu()
        return _tf.unsqueeze(-3).to(self.device)

    @property
    def theta(self):
        if self.hparams.use_covariate:
            theta = logit_to_distribution(self.lamb)
        else:
            theta = self.Lambda / self.Lambda.sum(0)
        return theta.float()

    @property
    def Chat(self):
        # if self.hparams.use_covariate:
        #     # Chat = (self.tf.squeeze(-3).transpose(-2, -3) * self.theta).sum(-1) * self.Y.sum(dim=(0, 1, 2, 3, 4, 5))
        #     # bias_correction = (self.tf * self.phi.mean(dim=-3, keepdim=True)).sum(-1)
        #     Chat = (self.tf.matmul(self.theta) * self.Y.sum(dim=(0, 1, 2, 3, 4, 5))).squeeze()
        #     #
        #     # if self.hparams.bias_correction:
        #     #     Chat *= self.bias_corrector
        return (self.tf.matmul(self.theta) * self.Y.sum(dim=(0, 1, 2, 3, 4, 5))).squeeze()

    @property
    def bias_corrector(self):
        return (self.tf.cpu() * self.phi.mean(dim=-3, keepdim=True)).sum(-1).transpose(-1, -2).to(self.device)

    @property
    def missing_rate(self) -> torch.Tensor:
        """
        Return : _m (2*2 tensor)
        _m[:,:] : missing rate of t and r
        """

        y_tr = self.Y.sum(dim=(2, 3, 4, -2, -1))

        _m00 = y_tr[:2, :2].sum(dim=(0, 1)).float() / y_tr.sum(dim=(0, 1))
        _m01 = y_tr[:2, 2].sum(dim=0).float() / y_tr.sum(dim=(0, 1))
        _m10 = y_tr[2, :2].sum(dim=0).float() / y_tr.sum(dim=(0, 1))
        _m11 = y_tr[2, 2].float() / y_tr.sum(dim=(0, 1))

        m = torch.stack([_m00, _m01, _m10, _m11]).reshape(2, 2)

        return m

    @property
    def tf_params(self):
        return [self._T0, self._t, self._r, self._e, self._n, self._c]

    @property
    def negative_elbo(self):
        TF = (self.T.cpu() * self.F.cpu()).unsqueeze(-3)

        Y = self.Y.transpose(-1, -2).unsqueeze(-1)

        neg_elbo = -(Y.cpu() * self.phi * torch.log(TF.clamp(1e-20))).sum().to(self.device)

        if self.hparams.use_covariate:

            tr = torch.inverse(self.Sigma_mat.float()).matmul(self.Delta.float())
            tr = -torch.diagonal(tr, dim1=1, dim2=2).sum() / 2

            neg_elbo -= tr

            EqGamma = (self.mu - self.lamb).T.float()
            EqGamma = EqGamma.matmul(torch.inverse(self.Sigma_mat).float())
            EqGamma = EqGamma.matmul((self.mu - self.lamb).float())
            EqGamma = torch.trace(EqGamma)

            x = self.X.T.matmul(self.zeta).matmul(self.X).float()
            x = torch.diagonal(x, dim1=1, dim2=2).sum(dim=1)
            x = x.float().dot(torch.diagonal(torch.inverse(self.Sigma_mat)).float())

            EqGamma += x

            neg_elbo += 0.5 * EqGamma

            log_det = torch.logdet(self.Sigma_mat + 0.01 * torch.eye(self.hparams.rank - 1, device=self.device)) \
                      - torch.logdet(self.Delta + + 0.01 * torch.eye(self.hparams.rank - 1, device=self.device))

            neg_elbo += 0.5 * log_det.sum()

            DivGamma = torch.diagonal(self.zeta, dim1=1, dim2=2).sum(dim=1)
            DivGamma /= self.sigma ** 2 + 1e-20
            DivGamma += (self.Xi ** 2).sum(axis=-1) / (self.sigma ** 2).clamp(1e-20)
            DivGamma += 2 * self.hparams.p * torch.log(self.sigma.clamp(1e-20))
            DivGamma -= torch.log(torch.det(self.zeta).clamp(1e-20))

            neg_elbo += 0.5 * DivGamma.sum()


        else:
            DivTheta = self.hparams.D * torch.lgamma(self.H + 1e-20).sum()

            DivTheta -= torch.lgamma(self.Lambda + 1e-20).sum()

            DivTheta += ((self.Lambda.T - self.H).T \
                         * (torch.digamma(self.Lambda + 1e-20) - 1)).sum()

            neg_elbo += DivTheta

        return neg_elbo.item() / self.hparams.D

    def configure_optimizers(self):
        return
