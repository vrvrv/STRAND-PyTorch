import os
import time
import h5py
import logging
import torch.nn as nn
import pytorch_lightning as pl

from .utils import *
from .functions import *
from .laplace_approximation import LaplaceApproximation
from .kappa import kappa_opt_model
from scipy.stats import entropy, poisson

from scipy.linalg import solve_sylvester

from tqdm import tqdm

logger = logging.getLogger('src.train')

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
                 init: str = 'nmf',
                 joint_init_iter=10,
                 nmf_max_iter: int = 10000,
                 e_iter: int = 20,
                 tf_lr: float = 0.1,
                 tf_max_steps: int = 200,
                 tf_lr_decay: float = 0.8,
                 rho_0: float = 10,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        with h5py.File(data_dir, "r") as f:
            if 'feature' in f.keys() and use_covariate is True:
                self.register_buffer(
                    'X', torch.tensor(f['feature'], dtype=torch.float), persistent=False
                )
                self.hparams['p'] = self.X.shape[0]
            else:
                self.X = None

            self.register_buffer(
                'Y', torch.tensor(f['count_tensor'], dtype=torch.float), persistent=False
            )
            self.hparams['V'], self.hparams['D'] = self.Y.size(-2), self.Y.size(-1)

        init = Initializer(
            rank=self.hparams.rank,
            k_dims=[t_dim, r_dim, e_dim, n_dim, c_dim],
            init=init,
            nmf_max_iter=nmf_max_iter
        )
        init.init(
            count_tensor=self.Y,
            feature=self.X
        )

        for param_name, param in init.buffers.items():
            self.register_buffer(param_name, param)

        self.la = LaplaceApproximation(
            max_iter=self.hparams.laplace_approx_conf.max_iter,
            batch_size=self.hparams.laplace_approx_conf.batch_size
        )

        self.m_step_kappa = kappa_opt_model(
            _kt=self._kt,
            _ki_t=self._ki_t, _ki_r=self._ki_r, _ki_e=self._ki_e, _ki_n=self._ki_n, _ki_c=self._ki_c,
            _kf_t=self._kf_t, _kf_r=self._kf_r, _kf_e=self._kf_e, _kf_n=self._kf_n, _kf_c=self._kf_c,
            _kc_t=self._kc_t, _kc_r=self._kc_r, _kc_e=self._kc_e, _kc_n=self._kc_n, _kc_c=self._kc_c
        )

        self.automatic_optimization = False

    def on_train_start(self) -> None:
        if self.hparams.use_covariate:
            self._cache_XXT = self.X.matmul(self.X.T)

    def e_step(self):
        if self.hparams.use_covariate:
            # UPDATE zeta
            for k in range(self.hparams.rank - 1):
                zeta_k = torch.inverse(
                    1 / (self.sigma[k] ** 2) * torch.eye(self.hparams.p, device=self.device) \
                    + self.Sigma_inv[k, k] * self._cache_XXT
                ).float()

                self.zeta[k] = zeta_k

            # Update Xi, eta and Delta
            for it in tqdm(range(self.hparams.e_iter), desc=f'E-STEP ({self.current_epoch})', leave=True):
                # Update Xi
                A = (self.Sigma_mat / (self.sigma ** 2)).cpu().numpy()
                B = self._cache_XXT.cpu().numpy()
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
                    eps=self.hparams.laplace_approx_conf.eps,
                    return_Delta=True
                )

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
                self.sigma[k] = sigma_k.clamp_(1e-8)
        else:
            self.H = self.Lambda.sum(-1) / self.Lambda.sum()

        with torch.no_grad():
            self.m_step_kappa._kt = nn.Parameter(self._kt)
            self.m_step_kappa._kc_t = nn.Parameter(self._kc_t)
            self.m_step_kappa._kc_r = nn.Parameter(self._kc_r)
            self.m_step_kappa._kc_e = nn.Parameter(self._kc_e)
            self.m_step_kappa._kc_n = nn.Parameter(self._kc_n)
            self.m_step_kappa._kc_c = nn.Parameter(self._kc_c)

            self.m_step_kappa._kf_t = nn.Parameter(self._kf_t)
            self.m_step_kappa._kf_r = nn.Parameter(self._kf_r)
            self.m_step_kappa._kf_e = nn.Parameter(self._kf_e)
            self.m_step_kappa._kf_n = nn.Parameter(self._kf_n)
            self.m_step_kappa._kf_c = nn.Parameter(self._kf_c)

            self.m_step_kappa._ki_t = nn.Parameter(self._ki_t)
            self.m_step_kappa._ki_r = nn.Parameter(self._ki_r)
            self.m_step_kappa._ki_e = nn.Parameter(self._ki_e)
            self.m_step_kappa._ki_n = nn.Parameter(self._ki_n)
            self.m_step_kappa._ki_c = nn.Parameter(self._ki_c)

        yphi = yphi_loader(
            Y=self.Y, phi_d=self.phi_d
        )

        self.m_step_kappa.fit(
            yphi_loader=yphi,
            lr=self.hparams.tf_lr,
            max_steps=self.hparams.tf_max_steps,
        )

        for k in ['kt', 'ki_t', 'ki_r', 'ki_e', 'ki_n', 'ki_c', 'kf_t', 'kf_r', 'kf_e', 'kf_n', 'kf_c', 'kc_t', 'kc_r', 'kc_e', 'kc_n', 'kc_c']:
            setattr(self, f"_{k}", getattr(self.m_step_kappa, f"_{k}"))

    def training_step(self, *args, **kwargs):
        self.e_step()
        self.m_step()
        nelbo = self.negative_elbo
        ll = self.log_likelihood

        self.log("negative_elbo", nelbo, prog_bar=True, logger=True)
        self.log("log_likelihood", float(ll), prog_bar=True, logger=True)

    @property
    def Sigma_inv(self):
        return torch.inverse(self.Sigma_mat).float()

    @property
    def mu(self):
        return self.Xi.matmul(self.X).float()

    @property
    def phi(self) -> torch.Tensor:
        if self.hparams.use_covariate:
            phi = Phi(
                beta=self.beta, lambda_or_Lambda=('lambda', self.lamb)
            )
        else:
            phi = Phi(
                beta=self.beta, lambda_or_Lambda=('Lambda', self.Lambda)
            )

        return phi.float()

    def phi_d(self, d):
        if self.hparams.use_covariate:
            phi_d = Phi_d(
                beta=self.beta, lambda_or_Lambda=('lambda', self.lamb[:, d])
            )

        else:
            phi_d = Phi_d(
                beta=self.beta, lambda_or_Lambda=('Lambda', self.Lambda[:, d])
            )
        return phi_d

    @property
    def Yphi(self):
        if self.hparams.use_covariate:
            yphi = Yphi(
                Y=self.Y, beta=self.beta, lambda_or_Lambda=('lambda', self.lamb)
            )
        else:
            yphi = Yphi(
                Y=self.Y, beta=self.beta, lambda_or_Lambda=('Lambda', self.Lambda)
            )
        return yphi

    @property
    def beta(self):
        return topic_word_dist(
            _kt=self._kt,
            _kc_t=self._kc_t,
            _kc_r=self._kc_r,
            _kc_e=self._kc_e,
            _kc_n=self._kc_n,
            _kc_c=self._kc_c,
            _ki_t=self._ki_t,
            _ki_r=self._ki_r,
            _ki_e=self._ki_e,
            _ki_n=self._ki_n,
            _ki_c=self._ki_c,
            _kf_t=self._kf_t,
            _kf_r=self._kf_r,
            _kf_e=self._kf_e,
            _kf_n=self._kf_n,
            _kf_c=self._kf_c
        )

    @property
    def theta(self):
        if self.hparams.use_covariate:
            theta = logit_to_distribution(self.lamb)
        else:
            theta = self.Lambda / self.Lambda.sum(0)
        return theta.float()

    @property
    def Chat(self):
        theta = self.theta
        phat = self.beta.float().matmul(theta)

        return (phat * self.Y.sum(dim=(0, 1, 2, 3, 4, 5))).squeeze()

    @property
    def negative_elbo(self):
        with torch.no_grad():
            beta = self.beta.cpu().unsqueeze(-3)

            Y = self.Y.transpose(-1, -2).unsqueeze(-1)

            neg_elbo = -(Y.cpu() * self.phi * torch.log(beta.clamp(1e-20))).sum().to(self.device)

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

                log_det = torch.logdet(self.Sigma_mat + 0.001 * torch.eye(self.hparams.rank - 1, device=self.device)) \
                          - torch.logdet(self.Delta + 0.001 * torch.eye(self.hparams.rank - 1, device=self.device))

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
                DivTheta += ((self.Lambda.T - self.H).T * (torch.digamma(self.Lambda + 1e-20) - 1)).sum()

                neg_elbo += DivTheta

        return neg_elbo.item() / self.hparams.D

    @property
    def log_likelihood(self):
        with torch.no_grad():
            Chat = self.Chat.detach().cpu().numpy()
            ll = poisson(Chat.flatten()).logpmf(self.Y.flatten().cpu().numpy()).sum()
        return ll

    @property
    def num_params(self):

        k = torch.prod(torch.tensor(self._kt.size())) \
            + torch.prod(torch.tensor(self._kc_t.size())) \
            + torch.prod(torch.tensor(self._kc_r.size())) \
            + torch.prod(torch.tensor(self._kc_e.size())) \
            + torch.prod(torch.tensor(self._kc_n.size())) \
            + torch.prod(torch.tensor(self._kc_c.size())) \
            + torch.prod(torch.tensor(self._ki_t.size())) \
            + torch.prod(torch.tensor(self._ki_r.size())) \
            + torch.prod(torch.tensor(self._ki_e.size())) \
            + torch.prod(torch.tensor(self._ki_n.size())) \
            + torch.prod(torch.tensor(self._ki_c.size()))

        if self.hparams.use_covariate:
            k += torch.prod(torch.tensor(self.lamb.size()))
        else:
            k += torch.prod(torch.tensor(self.Lambda.size()))

        return int(k)

    def configure_optimizers(self):
        print(self.trainer.log_dir)
        return
