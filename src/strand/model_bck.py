import os
import h5py
import pickle
import logging
import pytorch_lightning as pl

from .utils import *
from .functions import *
from .laplace_approximation import LaplaceApproximation
from .tnf import TF_opt_model

from scipy.linalg import solve_sylvester

from tqdm import tqdm

logger = logging.getLogger('src.train')


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
                 init_method: dict = {'T0': 'nmf', 'factors': 'nmf', 'Xi': 'sylvester', 'jointly': True},
                 joint_init_iter=10,
                 nmf_max_iter: int = 10000,
                 e_iter: int = 20,
                 tf_lr: float = 0.1,
                 tf_batch_size: int = 32,
                 tf_max_steps: int = 200,
                 tf_lr_decay: float = 0.8,
                 regularization_coef: float = 0.001,
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
            T0_init=init_method['T0'],
            factors_init=init_method['factors'],
            Xi_init=init_method['Xi'],
            init_jointly=init_method['jointly'],
            nmf_max_iter=nmf_max_iter
        )
        init.init(
            count_tensor=self.Y,
            feature=self.X,
            factor_names=['t', 'r', 'e', 'n', 'c'],
            joint_init_true=joint_init_iter
        )

        for param_name, param in init.buffers.items():
            self.register_buffer(param_name, param)

        self.la = LaplaceApproximation(
            max_iter=self.hparams.laplace_approx_conf.max_iter,
            batch_size=self.hparams.laplace_approx_conf.batch_size,
            beta=self._beta
        )

        self.tnf = TF_opt_model(
            Y=self.Y, _T0=self._T0, _tT=self._tT, _rT=self._rT, _eT=self._eT, _nT=self._nT, _cT=self._cT,
            _t=self._t, _r=self._r, _e=self._e, _n=self._n, _c=self._c, _at=self._at, _ar=self._ar,
            e_dim=self.hparams.e_dim, n_dim=self.hparams.n_dim, c_dim=self.hparams.c_dim,
            regularization_coef=self.hparams.regularization_coef
        )

        self.automatic_optimization = False

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
            for it in pbar:
                torch.save(self.theta.cpu(), os.path.join(self.trainer.log_dir, f'c_{self.current_epoch}_{it}.pt'))
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
                    eps=self.hparams.laplace_approx_conf.eps,
                    return_Delta=(it == self.hparams.e_iter - 1)
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
                self.sigma[k] = sigma_k.clamp_(0.001)
        else:
            self.H = self.Lambda.sum(-1) / self.Lambda.sum()

        with torch.no_grad():
            self.tnf._T0 = nn.Parameter(self._T0)
            self.tnf._t = nn.Parameter(self._t)
            self.tnf._r = nn.Parameter(self._r)
            self.tnf._e = nn.Parameter(self._e)
            self.tnf._n = nn.Parameter(self._n)
            self.tnf._c = nn.Parameter(self._c)
            self.tnf._at = nn.Parameter(self._at)
            self.tnf._ar = nn.Parameter(self._ar)

            self.tnf._tT = nn.Parameter(self._tT)
            self.tnf._rT = nn.Parameter(self._rT)
            self.tnf._eT = nn.Parameter(self._eT)
            self.tnf._nT = nn.Parameter(self._nT)
            self.tnf._cT = nn.Parameter(self._cT)

        yphi = get_yphi_dataloader(
            Y=self.Y, phi_d=self.phi_d, batch_size=self.hparams.tf_batch_size
        )

        self.tnf.fit(
            yphi_loader=yphi,
            lr=self.hparams.tf_lr * self.hparams.tf_lr_decay ** self.current_epoch,
            max_steps=self.hparams.tf_max_steps
        )

        self._T0 = self.tnf._T0

        for k in ['t', 'r', 'e', 'n', 'c', 'at', 'ar']:
            setattr(self, f"_{k}", getattr(self.tnf, f"_{k}"))

    def training_step(self, *args, **kwargs):
        self.e_step()
        self.m_step()

        nelbo = self.negative_elbo

        self.log("negative_elbo", nelbo, on_epoch=True, prog_bar=True, logger=True)

        # torch.save(self.theta.cpu(), os.path.join(self.trainer.log_dir, f'c_{self.current_epoch}.pt'))
        # torch.save(self.tf.sum((0, 1, 2, 3, 4, 5)).cpu(),
        #            os.path.join(self.trainer.log_dir, f'tf_{self.current_epoch}.pt'))
        print(os.path.join(self.trainer.log_dir, f'c_{self.current_epoch}.pt'))

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
            c_dim=self.hparams.c_dim
        )

    @property
    def F(self):
        return factors_to_F(
            _t=self._t,
            _r=self._r,
            _e=self._e,
            _n=self._n,
            _c=self._c,
            _at=self._at,
            _ar=self._ar,
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
        return (self.tf.matmul(self.theta) * self.Y.sum(dim=(0, 1, 2, 3, 4, 5))).squeeze()

    @property
    def tf_params(self):
        return [self._T0, self._t, self._r, self._e, self._n, self._c]

    @property
    def negative_elbo(self):
        TF = self.tf.cpu().unsqueeze(-3)

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

            DivTheta += ((self.Lambda.T - self.H).T * (torch.digamma(self.Lambda + 1e-20) - 1)).sum()

            neg_elbo += DivTheta

        return neg_elbo.item() / self.hparams.D

    @property
    def _beta(self):
        return min(1, self.hparams.laplace_approx_conf.beta + self.current_epoch / 20)

    def on_train_start(self) -> None:
        torch.save(self.tf.sum((0, 1, 2, 3, 4, 5)).cpu(),
                   os.path.join(self.trainer.log_dir, f'tf_start.pt'))


    def configure_optimizers(self):
        return
