import torch
import math
from torch import nn, optim
from .functions import *
from tqdm import tqdm


class TF_opt_model(nn.Module):

    def __init__(
            self,
            Y: torch.Tensor,
            _T0: torch.Tensor,
            _t: torch.Tensor,
            _r: torch.Tensor,
            _e: torch.Tensor,
            _n: torch.Tensor,
            _c: torch.Tensor,
            e_dim: int,
            n_dim: int,
            c_dim: int
    ):

        super(TF_opt_model, self).__init__()

        self.rank = _t.size(dim=-1)

        self.register_buffer(
            'Y', Y, persistent=False
        )

        # Declare parameters
        self._T0 = nn.Parameter(_T0)

        self._t = nn.Parameter(_t)
        self._r = nn.Parameter(_r)
        self._e = nn.Parameter(_e)
        self._n = nn.Parameter(_n)
        self._c = nn.Parameter(_c)

        self.e_dim = e_dim
        self.n_dim = n_dim
        self.c_dim = c_dim

        self.device = _t.device

    def forward(self, yphi, idx):

        T = stack(
            _T0=self._T0,
            _t=self._t,
            _r=self._r,
            e_dim=self.e_dim,
            n_dim=self.n_dim,
            c_dim=self.c_dim,
            rank=self.rank
        )

        F = factors_to_F(
            _t=self._t,
            _r=self._r,
            _e=self._e,
            _n=self._n,
            _c=self._c,
            e_dim=self.e_dim,
            n_dim=self.n_dim,
            c_dim=self.c_dim,
            missing_rate=self._missing_rate,
            rank=self.rank,
            reduction=True
        )

        return -(yphi * torch.log((T * F).clamp_(1e-20)).unsqueeze(-3)).mean(-1).sum()

    def fit(
            self,
            yphi_loader,
            lr: float,
            max_steps: int
    ):
        self.optim = optim.Adam(self.parameters(), lr=lr)

        # if not hasattr(self, 'optim'):
        #     self.optim = optim.Adam(self.parameters(), lr=lr)
        # else:
        #     self.optim.param_groups[0]['params'] = [p for p in self.parameters()]

        with tqdm(total=max_steps, desc='[M] T & F optimization', leave=False) as pbar:
            cur_step = 0
            for _ in range(math.ceil(max_steps/len(yphi_loader))):
                avg_loss = 0
                for i, (yphi_batch, idx) in enumerate(yphi_loader):
                    self.optim.zero_grad()
                    loss = self(yphi=yphi_batch, idx=idx)
                    loss.backward()

                    self.optim.step()

                    if cur_step >= max_steps:
                        break

                    avg_loss += loss.item()
                    pbar.update(1)

                    cur_step += 1

                pbar.set_postfix({'loss': avg_loss})

        self._T0.detach_()
        self._t.detach_()
        self._r.detach_()
        self._e.detach_()
        self._n.detach_()
        self._c.detach_()
    @property
    def _missing_rate(self) -> torch.Tensor:
        """
        Return : _m (2*2 tensor)
        _m[:,:] : missing rate of t and r
        """
        y_tr = self.Y.sum(dim=(2, 3, 4, -2, -1))

        _m00 = y_tr[:2, :2].sum(dim=(0, 1)).float() / y_tr.sum(dim=(0, 1))
        _m01 = y_tr[:2, 2].sum(dim=0).float() / y_tr.sum(dim=(0, 1))
        _m10 = y_tr[2, :2].sum(dim=0).float() / y_tr.sum(dim=(0, 1))
        _m11 = y_tr[2, 2].float() / y_tr.sum(dim=(0, 1))

        m = torch.stack([_m00, _m01, _m10, _m11])

        return m.reshape(2, 2)

    def missing_rate(self, idx) -> torch.Tensor:
        """
        Return : _m (2*2 tensor)
        _m[:,:] : missing rate of t and r
        """
        y_tr = self.Y.sum(dim=(2, 3, 4, -2))[..., idx]

        _m00 = y_tr[:2, :2].sum(dim=(0, 1)).float() / y_tr.sum(dim=(0, 1))
        _m01 = y_tr[:2, 2].sum(dim=0).float() / y_tr.sum(dim=(0, 1))
        _m10 = y_tr[2, :2].sum(dim=0).float() / y_tr.sum(dim=(0, 1))
        _m11 = y_tr[2, 2].float() / y_tr.sum(dim=(0, 1))

        m = torch.stack([_m00, _m01, _m10, _m11])

        return m.reshape(2, 2, -1)

#
# class TF_opt_model(nn.Module):
#
#     def __init__(
#             self,
#             Y: torch.Tensor,
#             _T0: torch.Tensor,
#             _t: torch.Tensor,
#             _r: torch.Tensor,
#             _e: torch.Tensor,
#             _n: torch.Tensor,
#             _c: torch.Tensor,
#             e_dim: int,
#             n_dim: int,
#             c_dim: int
#     ):
#
#         super(TF_opt_model, self).__init__()
#
#         self.D = Y.size(dim=-1)
#         self.rank = _t.size(dim=-1)
#
#         self.register_buffer(
#             'Y', Y.transpose(-1, -2).unsqueeze(-1)
#         )
#
#         # Declare parameters
#         self._T0 = nn.Parameter(_T0)
#
#         self._t = nn.Parameter(_t)
#         self._r = nn.Parameter(_r)
#         self._e = nn.Parameter(_e)
#         self._n = nn.Parameter(_n)
#         self._c = nn.Parameter(_c)
#
#         self.e_dim = e_dim
#         self.n_dim = n_dim
#         self.c_dim = c_dim
#
#     def forward(self, yphi):
#
#         T = stack(
#             _T0=self._T0,
#             _t=self._t,
#             _r=self._r,
#             e_dim=self.e_dim,
#             n_dim=self.n_dim,
#             c_dim=self.c_dim,
#             rank=self.rank
#         )
#
#         F = factors_to_F(
#             _t=self._t,
#             _r=self._r,
#             _e=self._e,
#             _n=self._n,
#             _c=self._c,
#             e_dim=self.e_dim,
#             n_dim=self.n_dim,
#             c_dim=self.c_dim,
#             missing_rate=self.missing_rate,
#             rank=self.rank
#         )
#
#         TF = T.unsqueeze(-3) * F.unsqueeze(-2)
#         return -(yphi * torch.log(TF + 1e-20)).mean(dim=(-1, -2)).sum()
#
#     def fit(
#             self,
#             phi,
#             lr: float,
#             max_iter: int
#     ):
#
#         yphi = self.Y * phi
#
#         if not hasattr(self, 'optim'):
#             self.optim = optim.AdamW(self.parameters(), lr=lr)
#         else:
#             self.optim.param_groups[0]['params'] = [p for p in self.parameters()]
#             # self.optim.add_param_group({'params': [p for p in self.parameters()]})
#
#         pbar = tqdm(range(max_iter), desc='[M] T & F optimization', leave=False)
#         for _ in pbar:
#             self.zero_grad()
#
#             loss = self(yphi)
#             loss.backward()
#
#             self.optim.step()
#             if loss.isnan().any():
#                 return
#
#             pbar.set_postfix({'loss': loss.item()})
#
#         self._T0.detach_()
#         self._t.detach_()
#         self._r.detach_()
#         self._e.detach_()
#         self._n.detach_()
#         self._c.detach_()
#
#     @property
#     def missing_rate(self):
#         y_tr = self.Y.sum(axis=(2, 3, 4, -2))
#
#         __m00 = y_tr[:2, :2].sum(axis=(0, 1)).float() / y_tr.sum(dim=(0, 1))
#         __m01 = y_tr[:2, 2].sum(axis=(0)).float() / y_tr.sum(dim=(0, 1))
#         __m10 = y_tr[2, :2].sum(axis=(0)).float() / y_tr.sum(dim=(0, 1))
#         __m11 = y_tr[2, 2].float() / y_tr.sum(dim=(0, 1))
#
#         missing_rate = torch.stack([__m00, __m01, __m10, __m11]).reshape(2, 2, -1)
#
#         return missing_rate
