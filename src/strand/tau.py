import torch
import math
from torch import nn, optim
from .functions import *
from tqdm import tqdm
from statistics import mean


class tau_opt(nn.Module):

    def __init__(
            self,
            _tc_t: torch.Tensor,
            _tc_r: torch.Tensor,
            _tc_e: torch.Tensor,
            _tc_n: torch.Tensor,
            _tc_c: torch.Tensor,
            regularization_coef: float,
            **kwargs
    ):

        super(tau_opt, self).__init__()

        # Decalre parameters

        self._c_t = nn.Parameter(_tc_t)
        self._kc_r = nn.Parameter(_tc_r)
        self._kc_e = nn.Parameter(_tc_e)
        self._kc_n = nn.Parameter(_tc_n)
        self._kc_c = nn.Parameter(_tc_c)

        self.regularization_coef = regularization_coef

        self.epoch = 0

    def forward(self, lamb, yphi):

        theta = topic_prev_dist(
            lamb=lamb,
            _tc_t=self._tc_t,
            _tc_r=self._tc_r,
            _tc_e=self._tc_e,
            _tc_n=self._tc_n,
            _tc_c=self._tc_c,
        ).float()

        loss = -(yphi.mean(-2) * torch.log(theta)).mean()

        reg_ct = torch.abs(self._tc_t).mean()
        reg_cr = torch.abs(self._tc_r).mean()
        reg_ce = torch.abs(self._tc_e).mean()
        reg_cn = torch.abs(self._tc_n).mean()
        reg_cc = torch.abs(self._tc_c).mean()

        reg = reg_ct + reg_cr + reg_ce + reg_cn + reg_cc

        return loss, self.regularization_coef * reg

    def fit(
            self,
            yphi_loader,
            lamb,
            lr: float,
            max_steps: int,
            **kwargs
    ):
        self.epoch += 1
        self.optim = optim.Adam(self.parameters(), lr=lr)
        # self.optim.param_groups[0]['params'].append(lamb)

        with tqdm(total=max_steps, desc=f'[{self.epoch}-th M-step] Tau optimization', leave=False) as pbar:
            cur_step = 0
            for _ in range(math.ceil(max_steps / len(yphi_loader))):
                avg_loss = []
                avg_reg = []
                for i, (yphi_batch, idx) in enumerate(yphi_loader):
                    self.optim.zero_grad()
                    loss, reg = self(yphi=yphi_batch, lamb=lamb[:, idx])
                    (loss + reg).backward()

                    self.optim.step()

                    if cur_step >= max_steps:
                        break

                    avg_loss.append(loss.item() * yphi_batch.size(-3))
                    avg_reg.append(reg.item() * yphi_batch.size(-3))
                    pbar.update(1)

                    cur_step += 1

                pbar.set_postfix(
                    {
                        'loss': mean(avg_loss) * 100,
                        'reg': mean(avg_reg)
                    }
                )

        self._tc_t.detach_()
        self._tc_r.detach_()
        self._tc_e.detach_()
        self._tc_n.detach_()
        self._tc_c.detach_()
