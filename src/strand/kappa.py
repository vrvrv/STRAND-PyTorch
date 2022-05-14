import math
from torch import nn, optim
from .functions import *
from tqdm import tqdm
from statistics import mean


class kappa_opt_model(nn.Module):

    def __init__(
            self,
            _kt: torch.Tensor,
            _kc_t: torch.Tensor,
            _kc_r: torch.Tensor,
            _kc_e: torch.Tensor,
            _kc_n: torch.Tensor,
            _kc_c: torch.Tensor,
            _kf_t: torch.Tensor,
            _kf_r: torch.Tensor,
            _kf_e: torch.Tensor,
            _kf_n: torch.Tensor,
            _kf_c: torch.Tensor,
            _ki_t: torch.Tensor,
            _ki_r: torch.Tensor,
            _ki_e: torch.Tensor,
            _ki_n: torch.Tensor,
            _ki_c: torch.Tensor,
            **kwargs
    ):

        super(kappa_opt_model, self).__init__()

        # Decalre parameters
        self._kt = nn.Parameter(_kt)
        self._kc_t = nn.Parameter(_kc_t)
        self._kc_r = nn.Parameter(_kc_r)
        self._kc_e = nn.Parameter(_kc_e)
        self._kc_n = nn.Parameter(_kc_n)
        self._kc_c = nn.Parameter(_kc_c)

        self._kf_t = nn.Parameter(_kf_t)
        self._kf_r = nn.Parameter(_kf_r)
        self._kf_e = nn.Parameter(_kf_e)
        self._kf_n = nn.Parameter(_kf_n)
        self._kf_c = nn.Parameter(_kf_c)

        self._ki_t = nn.Parameter(_ki_t)
        self._ki_r = nn.Parameter(_ki_r)
        self._ki_e = nn.Parameter(_ki_e)
        self._ki_n = nn.Parameter(_ki_n)
        self._ki_c = nn.Parameter(_ki_c)

        self.epoch = 0

    def forward(self, yphi):
        beta = topic_word_dist(
            _kt=self._kt,
            _ki_t=self._ki_t,
            _ki_r=self._ki_r,
            _ki_e=self._ki_e,
            _ki_n=self._ki_n,
            _ki_c=self._ki_c,
            _kc_t=self._kc_t,
            _kc_r=self._kc_r,
            _kc_e=self._kc_e,
            _kc_n=self._kc_n,
            _kc_c=self._kc_c,
            _kf_t=self._kf_t,
            _kf_r=self._kf_r,
            _kf_e=self._kf_e,
            _kf_n=self._kf_n,
            _kf_c=self._kf_c
        ).float()

        loss = -(yphi * (torch.log(beta))).mean()

        return loss

    def fit(
            self,
            yphi_loader,
            lr: float,
            max_steps: int,
            **kwargs
    ):
        self.epoch += 1
        self.optim = optim.Adamax(self.parameters(), lr=lr)

        with tqdm(total=max_steps, desc=f'[{self.epoch}-th M-step] Kappa optimization', leave=False) as pbar:
            cur_step = 0
            avg_loss = []
            for _ in range(math.ceil(max_steps)):
                self.optim.zero_grad()
                loss = self(yphi=yphi_loader)
                loss.backward()

                self.optim.step()

                if cur_step >= max_steps:
                    break

                avg_loss.append(loss.item())
                pbar.update(1)

                cur_step += 1

                pbar.set_postfix(
                    {
                        'loss': mean(avg_loss)
                    }
                )


        self._kt.detach_()
        self._kc_t.detach_()
        self._kc_r.detach_()
        self._kc_e.detach_()
        self._kc_n.detach_()
        self._kc_c.detach_()

        self._kf_t.detach_()
        self._kf_r.detach_()
        self._kf_e.detach_()
        self._kf_n.detach_()
        self._kf_c.detach_()

        self._ki_t.detach_()
        self._ki_r.detach_()
        self._ki_e.detach_()
        self._ki_n.detach_()
        self._ki_c.detach_()
