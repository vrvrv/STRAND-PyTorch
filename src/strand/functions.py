import torch
import numpy as np


def logit(tensor: torch.Tensor, eps=1e-20) -> torch.Tensor:
    """ Logit transformation

    Args:
        tensor (`torch.Tensor`): The input tensor.
        eps (: obj: `float`): The small value for numerical stability.

    Returns :
        logit(tensor)

    """
    denom = tensor[-1]
    denom[denom < eps] = eps

    odd_ratio = tensor[:-1] / denom
    odd_ratio[odd_ratio < eps] = eps

    if isinstance(odd_ratio, torch.Tensor):
        logit = torch.log(odd_ratio)
    elif isinstance(odd_ratio, np.ndarray):
        logit = torch.log(torch.from_numpy(odd_ratio).float())
    else:
        raise TypeError
    return logit


def logit_to_distribution(tensor: torch.Tensor) -> torch.Tensor:
    return torch.softmax(torch.cat([tensor, torch.zeros((1, tensor.size(dim=1)), device=tensor.device)], dim=0), dim=0)


def Phi(
        beta: torch.Tensor,
        lambda_or_Lambda: tuple,
        **kwargs
):
    name, lam = lambda_or_Lambda

    if name == 'lambda':
        lam = torch.cat([lam.T, torch.zeros((lam.size(1), 1), device=lam.device)], dim=1)
    elif name == 'Lambda':
        lam = torch.log(lam.clamp_(1e-10)).T

    phi = torch.log(beta).cpu().unsqueeze(-3) + lam.unsqueeze(-2).cpu()
    return torch.softmax(phi, dim=-1)

def Phi_d(
        beta,
        lambda_or_Lambda: tuple,
        **kwargs
):
    name, lam_d = lambda_or_Lambda

    if name == 'lambda':
        lam_d = torch.cat([lam_d, torch.tensor([0], device=lam_d.device)], dim=0)
    elif name == 'Lambda':
        lam_d = torch.log(lam_d.clamp_(min=1e-10)).T

    phi_d = torch.log(beta.clamp(1e-10)) + lam_d

    return torch.softmax(phi_d, dim=-1)


# def Yphi(
#         Y: torch.Tensor,
#         T: torch.Tensor,
#         F: torch.Tensor,
#         lambda_or_Lambda: tuple
# ) -> torch.Tensor:
#     device = Y.device
#
#     phi = Phi(T=T, F=F, lambda_or_Lambda=lambda_or_Lambda)
#     Y = Y.transpose(-1, -2).cpu()
#
#     yphi = Y.unsqueeze(-1) * phi
#
#     return yphi.sum(axis=(0, 1, 2, 3, 4, -2)).to(device)
def Yphi(
        Y: torch.Tensor,
        beta: torch.Tensor,
        lambda_or_Lambda: tuple
) -> torch.Tensor:
    device = Y.device

    phi = Phi(
        beta=beta,
        lambda_or_Lambda=lambda_or_Lambda
    )
    yphi = Y.transpose(-1, -2).cpu().unsqueeze(-1) * phi

    return yphi.sum(axis=(0, 1, 2, 3, 4, -2)).to(device)


def topic_prev_dist(
        lamb, _tc_t, _tc_r, _tc_e, _tc_n, _tc_c, **kwargs
):
    rank_1, D = lamb.size()

    t_dim = _tc_t.shape[0]
    r_dim = _tc_r.shape[0]
    e_dim = _tc_e.shape[0]
    n_dim = _tc_n.shape[0]
    c_dim = _tc_c.shape[0]

    lamb = lamb.T.reshape(1, 1, 1, 1, 1, D, rank_1)

    _tc_t = _tc_t.reshape(t_dim, 1, 1, 1, 1, D, 1)
    _tc_r = _tc_r.reshape(1, r_dim, 1, 1, 1, D, 1)
    _tc_e = _tc_e.reshape(1, 1, e_dim, 1, 1, D, 1)
    _tc_n = _tc_n.reshape(1, 1, 1, n_dim, 1, D, 1)
    _tc_c = _tc_c.reshape(1, 1, 1, 1, c_dim, D, 1)

    theta = lamb + _tc_t + _tc_r + _tc_e + _tc_n + _tc_c

    theta = torch.exp(
        torch.cat([theta, torch.zeros((t_dim, r_dim, e_dim, n_dim, c_dim, D, 1)).to(_tc_t.device)], dim=-1)
    )
    return theta / theta.sum(-1, keepdims=True)


def topic_word_dist(
        _kt,
        _ki_t,
        _ki_r,
        _ki_e,
        _ki_n,
        _ki_c,
        _kc_t,
        _kc_r,
        _kc_e,
        _kc_n,
        _kc_c,
        _kf_t,
        _kf_r,
        _kf_e,
        _kf_n,
        _kf_c,
        _wm=None,
        **kwargs):
    """

    :param _m: V
    :param _kt: K x V
    :param _kc_{m}: L_m x V
    :param _ki_{m}: L_m x V x K
    :param _k: L_m x K
    :return: Beta : L_t x ... x L_c x V x K
    """
    rank, V = _kt.size()

    t_dim = _ki_t.shape[0]
    r_dim = _ki_r.shape[0]
    e_dim = _ki_e.shape[0]
    n_dim = _ki_n.shape[0]
    c_dim = _ki_c.shape[0]

    wm = 1

    # if _wm is None:
    #     wm = 1
    # else:
    #     wm = torch.exp(_wm) / (1+torch.exp(_wm))

    # _m = _m.reshape(1, 1, 1, 1, 1, V, 1) * wm
    # _kt = _kt.reshape(1, 1, 1, 1, 1, V, rank)

    # _mkt = (_m + _kt).T.reshape(1, 1, 1, 1, 1, V, rank)
    _kt = _kt.T.reshape(1, 1, 1, 1, 1, V, rank)

    _kf_t = _kf_t.reshape(t_dim, 1, 1, 1, 1, 1, rank)
    _kf_r = _kf_r.reshape(1, r_dim, 1, 1, 1, 1, rank)
    _kf_e = _kf_e.reshape(1, 1, e_dim, 1, 1, 1, rank)
    _kf_n = _kf_n.reshape(1, 1, 1, n_dim, 1, 1, rank)
    _kf_c = _kf_c.reshape(1, 1, 1, 1, c_dim, 1, rank)

    _kc_t = _kc_t.reshape(t_dim, 1, 1, 1, 1, V, 1)
    _kc_r = _kc_r.reshape(1, r_dim, 1, 1, 1, V, 1)
    _kc_e = _kc_e.reshape(1, 1, e_dim, 1, 1, V, 1)
    _kc_n = _kc_n.reshape(1, 1, 1, n_dim, 1, V, 1)
    _kc_c = _kc_c.reshape(1, 1, 1, 1, c_dim, V, 1)

    # _ki_t = _ki_t.reshape(t_dim, 1, 1, 1, 1, V, rank)
    # _ki_r = _ki_r.reshape(1, r_dim, 1, 1, 1, V, rank)
    # _ki_e = _ki_e.reshape(1, 1, e_dim, 1, 1, V, rank)
    # _ki_n = _ki_n.reshape(1, 1, 1, n_dim, 1, V, rank)
    # _ki_c = _ki_c.reshape(1, 1, 1, 1, c_dim, V, rank)

    beta = _kt \
           + _kf_t + _kf_r + _kf_e + _kf_n + _kf_c \
           + _kc_t + _kc_r + _kc_e + _kc_n + _kc_c
           # + _ki_t + _ki_r + _ki_e + _ki_n + _ki_c \

    Beta = torch.exp(beta)
    return Beta / Beta.sum((0, 1, 2, 3, 4, -2), keepdims=True)
