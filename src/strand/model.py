import pickle
import logging
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from .functions import *
from .laplace_approximation import LaplaceApproximation
from .tnf import TF_opt_model

from scipy.linalg import solve_sylvester
from sklearn.decomposition import NMF, non_negative_factorization

from tqdm import tqdm
from typing import Tuple

logger = logging.getLogger('src.train')


def deviance(pred, true):
    d_ij = 2 * (pred * torch.log(pred / (true + 1e-8) + 1e-8) - pred + true)
    return d_ij.mean()


def update_zeta(
        Sigma_mat: torch.Tensor,
        sigma: torch.Tensor,
        rank: int,
        X: torch.Tensor
) -> torch.Tensor:
    p = X.size(0)
    zeta = torch.empty((rank - 1, p, p))

    SigmaInv = torch.inverse(Sigma_mat)

    for k in range(rank - 1):
        zeta_k = torch.inverse(
            1 / (sigma[k] ** 2) * torch.eye(p) + SigmaInv[k, k] * X.matmul(X.T)
        )

        zeta[k] = zeta_k

    return zeta


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
                 tf_max_iter: int = 200,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.use_covariate:
            with open(data_dir['X'], "rb") as f:
                self.register_buffer(
                    'X', torch.from_numpy(pickle.load(f)).float()
                )

                self.hparams['p'] = self.X.shape[0]

        with open(data_dir['Y'], "rb") as f:
            self.register_buffer(
                'Y', torch.from_numpy(pickle.load(f)).float()
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

        def fixed_NMF(X: np.array, W: np.array, H: np.array) -> Tuple[np.array]:
            w, h, _ = non_negative_factorization(
                X=X,
                n_components=self.hparams.rank,
                max_iter=self.hparams.nmf_max_iter,
                W=W,
                H=H,
                update_H=False,
                init='custom'
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

        T = nmf.fit_transform(self.Y.sum(axis=(0, 1, 2, 3, 4)) + 0.1)
        E = nmf.components_

        T, E = fixed_NMF(X=self.Y.sum(axis=(0, 1, 2, 3, 4)), W=T, H=E)

        theta = torch.from_numpy(E / (E.sum(axis=0) + 1e-8))

        # Fit T0_CL, T0_CG, T0_TL, T0_TG
        Y_cl = self.Y[0, 0].cpu().sum(axis=(0, 1, 2)).numpy().astype(np.float64) + 1e-2
        Y_cg = self.Y[0, 1].cpu().sum(axis=(0, 1, 2)).numpy().astype(np.float64) + 1e-2
        Y_tl = self.Y[1, 0].cpu().sum(axis=(0, 1, 2)).numpy().astype(np.float64) + 1e-2
        Y_tg = self.Y[1, 1].cpu().sum(axis=(0, 1, 2)).numpy().astype(np.float64) + 1e-2

        cl, _ = fixed_NMF(X=Y_cl, W=T, H=E)
        cg, _ = fixed_NMF(X=Y_cg, W=T, H=E)
        tl, _ = fixed_NMF(X=Y_tl, W=T, H=E)
        tg, _ = fixed_NMF(X=Y_tg, W=T, H=E)

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
            dim = getattr(self.hparams, f"{factor}_dim")
            f = np.empty((dim, self.hparams.rank))

            for l in range(dim):
                Y_l = self.Y.index_select(i, torch.tensor([l])) \
                    .sum(axis=(0, 1, 2, 3, 4)).double()

                f_l, _, __ = non_negative_factorization(
                    n_components=self.hparams.rank,
                    X=Y_l,
                    W=None,
                    H=E,
                    update_H=False,
                    init='nndsvd'
                )

                f[l] = f_l.sum(axis=0)

            self.register_buffer(
                f'_{factor}', logit(torch.from_numpy(f / f.sum(axis=0)))
            )

        logger.info("Initialized factors t, r, e, n, c")

        if self.hparams.use_covariate:
            logger.info("Initialize zeta")
            tmp = 0.1 * torch.rand(self.hparams.rank - 1, self.hparams.p, 2)
            self.register_buffer(
                'zeta', torch.eye(self.hparams.p, device=self.device) + tmp.matmul(tmp.transpose(-1, -2))
            )

            logger.info("Initialize sigma")
            self.register_buffer(
                'sigma', torch.ones(self.hparams.rank - 1)
            )

            logger.info("Intialize Sigma_mat")
            tmp = torch.randn(self.hparams.rank - 1, self.hparams.rank - 1, device=self.device)
            self.register_buffer(
                'Sigma_mat', torch.eye(self.hparams.rank - 1, device=self.device) \
                             + tmp.matmul(tmp.T)
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
                'H', self.Lambda.mean(axis=1)
            )

        logger.info("Initialization Ended")

    def _random_init(self) -> dict:

        output = dict()

        _cl = logit(
            Dirichlet(torch.ones(self.V)).sample()
        )
        _cg = logit(
            Dirichlet(torch.ones(self.V)).sample()
        )
        _tl = logit(
            Dirichlet(torch.ones(self.V)).sample()
        )
        _tg = logit(
            Dirichlet(torch.ones(self.V)).sample()
        )

        _T0 = torch.stack([_cl, _cg, _tl, _tg]).reshape(2, 2, self.V - 1, self.hparams.rank)
        output['_T0'] = _T0
        self.logger.info("Initialized T0")

        _theta = logit(
            Dirichlet(torch.ones((self.D, self.V))).sample().T
        )

        for i, factor in enumerate(['t', 'r', 'e', 'n', 'c']):
            self.logger.info(f"Intializing factor, {factor}")
            dim = getattr(self.hparams, f"{factor}_dim")

            _f = logit(
                Dirichlet(torch.ones((self.hparams.rank, dim))).sample().T
            )

            output[f'_{factor}'] = _f

        self.logger.info("Initialized factors t, r, e, n, c")

        if self.X is not None:
            self.logger.info("Initialize zeta")
            tmp = 0.1 * torch.rand(self.hparams.rank - 1, self.p, 2)
            output['zeta'] = torch.eye(self.p) + tmp.matmul(tmp.transpose(-1, -2))

            self.logger.info("Initialize sigma")
            output['sigma'] = torch.ones(self.hparams.rank - 1)

            self.logger.info("Intialize Sigma_mat")
            tmp = torch.randn(self.hparams.rank - 1, self.hparams.rank - 1)
            output['Sigma_mat'] = torch.eye(self.hparams.rank - 1, dtype=torch.double) * 5 \
                                  + tmp.matmul(tmp.T)

            self.logger.info("Initialize Delta")
            output['Delta'] = output['Sigma_mat'].repeat((self.D, 1, 1))

            self.logger.info("Initialize lambda")
            output['lambda'] = _theta

            self.logger.info("Initialize Xi")
            output['Xi'] = torch.normal(0, 1, size=(self.rank - 1, self.p)).double()

        else:
            self.logger.info("Initialize Lambda")
            output['Lambda'] = logit_to_distribution(_theta)

            self.logger.info("Initialize H")
            output['H'] = output['Lambda'].mean(axis=1)

        self.logger.info("Initialization Ended")
        return output

    def e_step(self):

        if self.hparams.use_covariate:
            # UPDATE ZETA
            for k in range(self.hparams.rank - 1):
                self.zeta[k] = torch.inverse(
                    1 / (self.sigma[k] ** 2) * torch.eye(self.hparams.p, device=self.device) \
                    + self.Sigma_inv[k, k] * self.X.matmul(self.X.T)
                )

            # Update Xi, eta and Delta
            pbar = tqdm(range(self.hparams.e_iter), desc=f'E-STEP {self.current_epoch}', leave=False)
            for _ in pbar:
                # Update Xi
                A = (self.Sigma_mat / (self.sigma ** 2)).cpu().numpy()
                B = (self.X.matmul(self.X.T)).cpu().numpy()
                Q = (self.lamb.float().matmul(self.X.T)).cpu().numpy()

                self.Xi = torch.from_numpy(
                    solve_sylvester(A, B, Q)
                ).to(self.device)

                # Update eta and Delta
                with torch.no_grad():
                    self.la.eta = nn.Parameter(self.lamb)

                self.lamb, self.Delta = self.la.fit(
                    eta_init=self.lamb,
                    mu=self.mu,
                    Yphi=self.Yphi,
                    Sigma_inv=self.Sigma_inv,
                    lr=self.hparams.laplace_approx_conf.lr
                )
                pbar.set_postfix({'negative_elbo': self.negative_elbo})

        else:
            pass

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

            self.Sigma_mat = 0.01 * torch.diagonal(Sigma_mat) + 0.99 * Sigma_mat

            # Update sigma
            for k in range(self.hparams.rank - 1):
                self.sigma[k] = torch.sqrt((torch.trace(self.zeta[k]) + (self.Xi[k] ** 2).sum()) / self.hparams.p)

        with torch.no_grad():
            self.tnf._T0 = nn.Parameter(self._T0)
            self.tnf._t = nn.Parameter(self._t)
            self.tnf._r = nn.Parameter(self._r)
            self.tnf._e = nn.Parameter(self._e)
            self.tnf._n = nn.Parameter(self._n)
            self.tnf._c = nn.Parameter(self._c)

        self.tnf.fit(
            phi=self.phi,
            lr=self.hparams.tf_lr,
            max_iter=self.hparams.tf_max_iter
        )

        self._T0 = 0.99 * self.tnf._T0 + 0.01 * self._T0

        for k in ['t', 'r', 'e', 'n', 'c']:
            setattr(self, f"_{k}", getattr(self.tnf, f"_{k}"))

    def training_step(self, *args, **kwargs):
        self.e_step()
        self.m_step()

        self.log("negative_elbo", self.negative_elbo, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, *args, **kwargs):
        dev = deviance(self.Chat, self.Y)
        self.log("deviance", dev, logger=True)

    @property
    def Sigma_inv(self):
        return torch.inverse(self.Sigma_mat).double()

    @property
    def mu(self):
        return self.Xi.matmul(self.X)

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
        if self.X is not None:
            phi = Phi(
                T=self.T, F=self.F, lambda_or_Lambda=('lambda', self.lamb)
            )
        else:
            phi = Phi(
                T=self.T, F=self.F, lambda_or_Lambda=('Lambda', self.Lambda)
            )

        return phi

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
    def Chat(self):
        if self.hparams.use_covariate:
            # theta = logit_to_distribution(self.lamb).T
            # tf = self.T.unsqueeze(-3) * self.F.unsqueeze(-2)
            # Chat = (tf.transpose(-2, -3) * theta).sum(-1) * self.Y.sum(dim=(0, 1, 2, 3, 4, 5))
            # bias_correction = (tf.mean(dim=-3, keepdim=True) * self.phi.mean(dim=-3, keepdim=True)).sum(-1)

            tf = (self.T.unsqueeze(-3) * self.F.unsqueeze(-2)).mean(dim=-3, keepdim=True)
            theta = logit_to_distribution(self.lamb).float()

            p_lv_d = tf.matmul(theta)

            Chat = (p_lv_d * self.Y.sum((0, 1, 2, 3, 4, 5))).squeeze()

            bias_correction = (tf * self.phi.mean(dim=-3, keepdim=True)).sum(-1)

            return Chat * bias_correction.transpose(-1, -2)



    @property
    def missing_rate(self) -> torch.Tensor:
        """
        Return : _m (2*2 tensor)
        _m[:,:] : missing rate of t and r
        """
        y_tr = self.Y.sum(axis=(2, 3, 4, -2))

        __m00 = y_tr[:2, :2].sum(axis=(0, 1)).float() / y_tr.sum(dim=(0, 1))
        __m01 = y_tr[:2, 2].sum(axis=(0)).float() / y_tr.sum(dim=(0, 1))
        __m10 = y_tr[2, :2].sum(axis=(0)).float() / y_tr.sum(dim=(0, 1))
        __m11 = y_tr[2, 2].float() / y_tr.sum(dim=(0, 1))

        missing_rate = torch.stack([__m00, __m01, __m10, __m11]).reshape(2, 2, -1)

        return missing_rate

    @property
    def tf_params(self):
        return [self._T0, self._t, self._r, self._e, self._n, self._c]

    @property
    def negative_elbo(self):

        TF = self.T.unsqueeze(-3) * self.F.unsqueeze(-2)

        Y = self.Y.transpose(-1, -2).unsqueeze(-1)
        phi = self.phi

        neg_elbo = -(Y * phi * torch.log(TF + 1e-20)).sum()

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

            log_det = torch.log(torch.det(self.Sigma_mat) + 1e-20) \
                      - torch.log(torch.det(self.Delta) + 1e-20)

            neg_elbo += 0.5 * log_det.sum()

            DivGamma = torch.diagonal(self.zeta, dim1=1, dim2=2).sum(dim=1)
            DivGamma /= self.sigma ** 2 + 1e-20
            DivGamma += (self.Xi ** 2).sum(axis=-1) / (self.sigma ** 2 + 1e-20)
            DivGamma += 2 * self.hparams.p * torch.log(self.sigma + 1e-20)
            DivGamma -= torch.log(torch.det(self.zeta) + 1e-20)

            neg_elbo += 0.5 * DivGamma.sum()


        else:

            DivTheta = self.D * torch.lgamma(self.H + 1e-20).sum()

            DivTheta -= torch.lgamma(self.Lambda + 1e-20).sum()

            DivTheta += ((self.Lambda.T - self.H).T \
                         * (torch.digamma(self.Lambda + 1e-20) - 1)).sum()

            neg_elbo += DivTheta

        return neg_elbo.item() / self.hparams.D

    def configure_optimizers(self):
        return
