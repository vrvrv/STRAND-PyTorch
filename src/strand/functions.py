import torch


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

    return torch.log(odd_ratio)


def logit_to_distribution(tensor: torch.Tensor) -> torch.Tensor:
    return torch.softmax(torch.cat([tensor, torch.zeros((1, tensor.size(dim=1)), device=tensor.device)], dim=0), dim=0)


def Phi(
        T: torch.Tensor,
        F: torch.Tensor,
        lambda_or_Lambda: tuple
):
    name, lam = lambda_or_Lambda

    if name == 'lambda':
        lam = torch.cat([lam.T, torch.zeros((lam.size(1), 1), device=lam.device)], dim=1)
    elif name == 'Lambda':
        lam = torch.log(lam + 1e-20).T

    phi = torch.log(T.clamp(1e-10)).cpu()
    phi = phi.unsqueeze(-3) + lam.unsqueeze(-2).cpu()
    phi += torch.log(F.clamp(1e-10)).unsqueeze(-2).cpu()

    return torch.softmax(phi, dim=-1)


def Phi_d(
        T: torch.Tensor,
        F: torch.Tensor,
        lambda_or_Lambda: tuple
):
    name, lam_d = lambda_or_Lambda

    if name == 'lambda':
        lam_d = torch.cat([lam_d, torch.tensor([0], device=lam_d.device)], dim=0)
    elif name == 'Lambda':
        lam_d = torch.log(lam_d.clamp_(min=1e-10)).T

    phi_d = torch.log(T.clamp_(min=1e-10)) + torch.log(F.clamp_(min=1e-10)) + lam_d

    return torch.softmax(phi_d, dim=-1)


def Yphi(
        Y: torch.Tensor,
        T: torch.Tensor,
        F: torch.Tensor,
        lambda_or_Lambda: tuple
) -> torch.Tensor:
    device = Y.device

    phi = Phi(T=T, F=F, lambda_or_Lambda=lambda_or_Lambda)
    Y = Y.transpose(-1, -2).cpu()

    yphi = Y.unsqueeze(-1) * phi

    return yphi.sum(axis=(0, 1, 2, 3, 4, -2)).to(device)


def stack(_T0: torch.Tensor, _t: torch.Tensor, _r: torch.Tensor, e_dim: int, n_dim: int, c_dim: int, rank: int):
    [[_cl, _cg], [_tl, _tg]] = _T0

    cl = logit_to_distribution(_cl)
    cg = logit_to_distribution(_cg)
    tl = logit_to_distribution(_tl)
    tg = logit_to_distribution(_tg)

    t = logit_to_distribution(_t)
    r = logit_to_distribution(_r)

    V, K = _T0.size()[-2:]
    T = torch.empty((3, 3, e_dim, n_dim, c_dim, V + 1, K), device=_T0.device)

    t0 = t[0]
    r0 = r[0]

    c_ = r0 * cl + (1 - r0) * cg
    t_ = r0 * tl + (1 - r0) * tg
    _l = t0 * cl + (1 - t0) * tl
    _g = t0 * cg + (1 - t0) * tg

    __ = t0 * r0 * cl + t0 * (1 - r0) * cg + (1 - t0) * r0 * tl + (1 - t0) * (1 - r0) * tg

    T[0, 0] = cl
    T[1, 0] = tl
    T[2, 0] = _l
    T[0, 1] = cg
    T[1, 1] = tg
    T[2, 1] = _g
    T[0, 2] = c_
    T[1, 2] = t_
    T[2, 2] = __

    return T


def factors_to_F(
        _t: torch.Tensor,
        _r: torch.Tensor,
        _e: torch.Tensor,
        _n: torch.Tensor,
        _c: torch.Tensor,
        e_dim: int,
        n_dim: int,
        c_dim: int,
        rank: int,
        missing_rate: torch.Tensor,
        index=None,
        device=None,
        uniform_missing_rate=False
) -> torch.Tensor:

    if device == 'cpu':
        _t = _t.cpu()
        _r = _r.cpu()
        _e = _e.cpu()
        _n = _n.cpu()
        _c = _c.cpu()

        missing_rate = missing_rate.cpu()

        if isinstance(index, torch.Tensor):
            index = index.cpu()

    t = logit_to_distribution(_t)
    r = logit_to_distribution(_r)
    e = logit_to_distribution(_e)
    n = logit_to_distribution(_n)
    c = logit_to_distribution(_c)

    if not uniform_missing_rate:
        sample_size = missing_rate.size(-1)

        if index is None:
            index = torch.arange(0, sample_size)
            F = torch.ones((3, 3, e_dim, n_dim, c_dim, sample_size, rank), device=_t.device)
        else:
            F = torch.ones((3, 3, e_dim, n_dim, c_dim, len(index), rank), device=_t.device)

        for i in range(2):
            for j in range(2):
                F[i, j] *= t[i] * r[j] * missing_rate[0, 0][index, None]

            F[i, 2] *= t[i] * missing_rate[0, 1][index, None]

        for j in range(2):
            F[2, j] *= r[j] * missing_rate[1, 0][index, None]

        F[2, 2] *= missing_rate[1, 1][index, None]

    else:
        missing_rate = missing_rate.mean(-1)
        F = torch.ones((3, 3, e_dim, n_dim, c_dim, 1, rank), device=_t.device)

        for i in range(2):
            for j in range(2):
                F[i, j] *= t[i] * r[j] * missing_rate[0, 0]

            F[i, 2] *= t[i] * missing_rate[0, 1]

        for j in range(2):
            F[2, j] *= r[j] * missing_rate[1, 0]

        F[2, 2] *= missing_rate[1, 1]

    for l in range(e_dim):
        F[:, :, l] *= e[l]
    for l in range(n_dim):
        F[:, :, :, l] *= n[l]
    for l in range(c_dim):
        F[:, :, :, :, l] *= c[l]

    return F
