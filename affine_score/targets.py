"""Target densities, each with an exact sampler.

A target exposes `dim`, `dtype`, `log_prob(x)` of shape (..., 1) (up to a
constant), `grad_log_prob(x)`, `log_prob_max()` and `sample(n)`. The samplers
only ever call log_prob (importance sampling, ZOD-MC) or grad_log_prob (RDMC and
the exact endpoint of the post-processor); ZOD-MC also needs the maximum of
log_prob for its rejection step, and `sample` is used for the reference draws.

Targets with a dimension knob take `dim`; the others are fixed and accept
`dim=None` so every entry of `TARGETS` is built the same way. Defaults are the
paper's settings.
"""
import math

import torch


def _maximize(log_prob, x0, max_iter=200):
    """max log_prob, by L-BFGS from x0, for targets without a closed-form mode."""
    x = x0.detach().clone().requires_grad_()
    opt = torch.optim.LBFGS([x], max_iter=max_iter, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = -log_prob(x).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_prob(x.detach()).max())


# --------------------------------------------------------------------- Gaussian targets

class LogConcave:
    """N(0, diag(λ)) with λ log-spaced over [1, kappa]: the log-concave target, with
    `kappa` its condition number."""

    def __init__(self, dim=10, kappa=100.0, device='cpu'):
        self.dim, self.device, self.dtype = dim, device, torch.float32
        self.var = (torch.logspace(0, math.log10(kappa), dim) if kappa > 1
                    else torch.ones(dim)).to(device, self.dtype)

    def log_prob(self, x):
        return (-0.5 * (x ** 2 / self.var).sum(-1, keepdim=True))

    def grad_log_prob(self, x):
        return -x / self.var

    def log_prob_max(self):
        return 0.0

    def sample(self, n):
        return torch.randn(n, self.dim, device=self.device, dtype=self.dtype) * self.var.sqrt()


class GaussianMixture:
    """Σ_k w_k N(μ_k, Σ_k), from stacked tensors. Isotropic components (Σ_k = v_k I)
    take a cheaper path that never forms the (K, M, d) difference tensor."""

    def __init__(self, weights, means, covs, device='cpu', dtype=torch.float32):
        weights, means, covs = (torch.as_tensor(t, dtype=torch.float64) for t in (weights, means, covs))
        d = means.shape[1]
        self.dim, self.device, self.dtype = d, device, dtype
        self.means = means.to(device, dtype)
        self.chol = torch.linalg.cholesky(covs).to(device, dtype)
        self.prec = torch.linalg.inv(covs).to(device, dtype)
        self.log_norm = (torch.log(weights / weights.sum())
                         - 0.5 * (d * math.log(2 * math.pi) + torch.logdet(covs))).to(device, dtype)
        v = covs.diagonal(dim1=-2, dim2=-1).mean(-1)
        iso = torch.allclose(covs, v[:, None, None] * torch.eye(d, dtype=covs.dtype))
        self.iso_prec = (1.0 / v).to(device, dtype) if iso else None
        self._max = None

    def _components(self, x):
        """log(w_k N_k(x)) of shape (K, M) and Σ_k^{-1}(x - μ_k) of shape (K, M, d), in the
        wider of the query's and the target's dtype."""
        dt = torch.promote_types(x.dtype, self.dtype)
        flat, means, log_norm = x.reshape(-1, self.dim).to(dt), self.means.to(dt), self.log_norm.to(dt)
        if self.iso_prec is not None:
            quad = ((flat * flat).sum(-1)[None] - 2 * (means @ flat.T)
                    + (means * means).sum(-1)[:, None]) * self.iso_prec.to(dt)[:, None]
            return log_norm[:, None] - 0.5 * quad, None
        diff = flat[None] - means[:, None]
        px = torch.einsum('kij,kmj->kmi', self.prec.to(dt), diff)
        return log_norm[:, None] - 0.5 * (px * diff).sum(-1), px

    def log_prob(self, x):
        log_w, _ = self._components(x)
        return torch.logsumexp(log_w, dim=0).reshape(*x.shape[:-1], 1)

    def grad_log_prob(self, x):
        log_w, px = self._components(x)
        r = torch.softmax(log_w, dim=0)                                      # (K, M)
        if px is None:                                                       # Σ_k r_k p_k (x - μ_k)
            rp = r * self.iso_prec.to(r.dtype)[:, None]
            flat = x.reshape(-1, self.dim).to(r.dtype)
            out = flat * rp.sum(0)[:, None] - rp.T @ self.means.to(r.dtype)
        else:
            out = (r.unsqueeze(-1) * px).sum(0)
        return (-out).reshape(x.shape)

    def log_prob_max(self):
        """The best component mean, refined numerically."""
        if self._max is None:
            lp = self.log_prob(self.means).reshape(-1)
            self._max = _maximize(self.log_prob, self.means[lp.argmax()][None])
        return self._max

    def sample(self, n):
        w = torch.exp(self.log_norm + 0.5 * (self.dim * math.log(2 * math.pi)
                                             + 2 * torch.log(self.chol.diagonal(dim1=-2, dim2=-1)).sum(-1)))
        k = torch.multinomial(w, n, replacement=True)
        z = torch.randn(n, self.dim, 1, device=self.device, dtype=self.dtype)
        return self.means[k] + (self.chol[k] @ z).squeeze(-1)


class XGMM(GaussianMixture):
    """Two zero-mean Gaussians whose covariances are mirror images of each other.

    Coordinates come in dim/2 independent 2D blocks. In component 1 every block
    has covariance R(angle) diag(a^2, b^2) R(angle)^T and in component 2 the same
    with -angle, so the two components differ only in orientation: a sampler has
    to recover the X, not just the spread.
    """

    def __init__(self, dim=8, a=3.0, b=0.3, angle_deg=45.0, device='cpu'):
        assert dim % 2 == 0, 'XGMM is built from 2D blocks'
        covs = []
        for sign in (1.0, -1.0):
            th = math.radians(sign * angle_deg)
            R = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]],
                             dtype=torch.float64)
            block = R @ torch.diag(torch.tensor([a ** 2, b ** 2], dtype=torch.float64)) @ R.T
            covs.append(torch.block_diag(*[block] * (dim // 2)))
        super().__init__([0.5, 0.5], torch.zeros(2, dim), torch.stack(covs), device)

    def log_prob_max(self):
        return float(self.log_prob(torch.zeros(1, self.dim, device=self.device, dtype=self.dtype)))

    def sample(self, n):
        k = (torch.rand(n, device=self.device) < 0.5).long()
        z = torch.randn(n, self.dim, 1, device=self.device, dtype=self.dtype)
        return (self.chol[k] @ z).squeeze(-1)


class RandomGMM(GaussianMixture):
    """`num_modes` isotropic modes of width `sigma` at uniform positions in
    [0, 2 region]^dim. float64: at this width most points are far from every mode
    in float32."""

    def __init__(self, dim=10, num_modes=40, region=1.0, sigma=0.05, seed=0, device='cpu'):
        g = torch.Generator().manual_seed(seed)
        means = torch.rand(num_modes, dim, generator=g, dtype=torch.float64) * (2 * region)
        covs = (sigma ** 2) * torch.eye(dim, dtype=torch.float64).expand(num_modes, dim, dim)
        super().__init__(torch.full((num_modes,), 1.0 / num_modes), means, covs, device, torch.float64)


class TwoModeGMM(GaussianMixture):
    """Two unit Gaussians in 2D at (sep, sep) and (-sep/2, -sep/2) with weights 0.7 / 0.3:
    unequal weights and asymmetric placement, so a collapsed sampler is visibly biased."""

    def __init__(self, sep=5.0, dim=None, device='cpu'):
        assert dim in (None, 2)
        means = torch.tensor([[sep, sep], [-sep / 2, -sep / 2]])
        super().__init__([0.7, 0.3], means, torch.eye(2).expand(2, 2, 2), device)




class ZODMCGMM(GaussianMixture):
    """The four-component 2D mixture of the ZOD-MC paper, with its modes scaled so the
    farthest sits at distance `sep` (sep = 11 is the original)."""
    MEANS = torch.tensor([[0., 0.], [0., 11.], [9., 9.], [11., 0.]])
    COVS = torch.tensor([[[1.0, 0.5], [0.5, 1.0]], [[0.3, -0.2], [-0.2, 0.3]],
                         [[1.0, 0.3], [0.3, 1.0]], [[1.2, -1.0], [-1.0, 1.2]]])

    def __init__(self, sep=11.0, dim=None, device='cpu'):
        assert dim in (None, 2)
        super().__init__([.1, .2, .3, .4], self.MEANS * sep / 11, self.COVS, device)


def _rotation(dim, seed):
    """A fixed dense rotation, seeded on the CPU so the target is identical on every device."""
    g = torch.Generator().manual_seed(seed)
    Q, R = torch.linalg.qr(torch.randn(dim, dim, generator=g, dtype=torch.float64))
    return Q * torch.sign(torch.diagonal(R))[None]


def _cube_vertices(n, dim, seed):
    """`n` even-parity vertices of {-1, +1}^dim chosen by farthest-point greedy, so the
    closest pair is as far apart as the count allows."""
    assert 2 <= n <= 2 ** min(dim - 1, 62), f'the even-parity half of the {dim}-cube has {2 ** (dim - 1)} vertices'
    g = torch.Generator().manual_seed(seed)
    if dim <= 20:
        bits = (torch.arange(2 ** dim)[:, None] >> torch.arange(dim)) & 1
        even = bits[bits.sum(1) % 2 == 0].double() * 2 - 1
    else:   # a pool of random even-parity corners; in high dimension they are nearly equidistant
        pool = torch.randint(0, 2, (4096, dim), generator=g).double() * 2 - 1
        pool[(pool < 0).sum(1) % 2 == 1, -1] *= -1
        even = torch.unique(pool, dim=0)
    chosen = [int(torch.randint(even.shape[0], (1,), generator=g))]
    dmin = torch.cdist(even, even[chosen[0]:chosen[0] + 1]).squeeze(1)
    for _ in range(n - 1):
        chosen.append(int(dmin.argmax()))
        dmin = torch.minimum(dmin, torch.cdist(even, even[chosen[-1]:chosen[-1] + 1]).squeeze(1))
    return even[chosen]


class VertexGMM(GaussianMixture):
    """The mode-hopping target: `n_modes` Gaussians at vertices of a hypercube, rotated,
    with the closest pair at distance edge·√2.

    `rho` sets geometric weights w_k ∝ rho^k (`weights='linear'` gives n_modes − k);
    `narrow` shrinks the first `n_narrow` modes (default half of them) by that factor;
    `kappa` gives every mode its own anisotropic covariance of condition number kappa
    and unit determinant; `offset` shifts all modes along a fixed random direction by
    that fraction of their mean norm; `vol_spread` < 1 spreads the mode widths so their
    volumes range over that factor, with mode 0 the widest.
    """

    def __init__(self, dim=10, n_modes=8, edge=6.0 * math.sqrt(2.0), rho=1.0, weights='geometric',
                 narrow=1.0, n_narrow=None, kappa=1.0, offset=0.0, vol_spread=1.0, rot_seed=0,
                 device='cpu'):
        assert 0 < rho <= 1 and 0 < narrow <= 1 and kappa >= 1 and 0 <= offset < 1 and 0 < vol_spread <= 1
        n_narrow = (n_modes // 2 if n_narrow is None else n_narrow) if narrow < 1 else 0
        verts = _cube_vertices(n_modes, dim, rot_seed)
        dv = torch.cdist(verts, verts)
        means = verts * (edge * math.sqrt(2.0) / float(dv[~torch.eye(n_modes, dtype=torch.bool)].min()))
        means = means @ _rotation(dim, rot_seed).T
        if offset > 0:
            u = torch.randn(dim, generator=torch.Generator().manual_seed(rot_seed + 97), dtype=torch.float64)
            means = means + offset * float(means.norm(dim=1).mean()) * u / u.norm()
        k = torch.arange(n_modes, dtype=torch.float64)
        w = rho ** k if weights == 'geometric' else (n_modes - k)
        covs = torch.eye(dim, dtype=torch.float64).expand(n_modes, dim, dim).clone()
        if kappa > 1:
            half = math.log10(kappa) / 2
            lam = torch.logspace(-half, half, dim, dtype=torch.float64)          # det = 1
            for i in range(n_modes):
                Q = _rotation(dim, rot_seed + 1 + i)
                S = Q @ torch.diag(lam) @ Q.T
                covs[i] = 0.5 * (S + S.T)
        covs[:n_narrow] *= narrow ** 2
        if vol_spread < 1:
            s = torch.linspace(0, 1, n_modes, dtype=torch.float64)
            order = [0] + (1 + torch.randperm(n_modes - 1, generator=torch.Generator().manual_seed(rot_seed + 131))).tolist()
            s_of = torch.empty(n_modes, dtype=torch.float64)
            s_of[torch.tensor(order)] = s                                       # mode 0 gets s = 0: widest
            covs = covs * (vol_spread ** (2 * s_of / dim)).view(-1, 1, 1)
        super().__init__(w, means, covs, device, torch.float64)


# ------------------------------------------------------------------- curved targets

class Banana:
    """Haario et al.'s banana: x1 ~ N(0, s1^2), x2 | x1 ~ N(b (x1^2 - s1^2), 1),
    x3..xd ~ N(0, 1)."""

    def __init__(self, dim=10, b=0.1, s1=3.0, device='cpu'):
        assert dim >= 2
        self.dim, self.b, self.s1, self.device = dim, b, s1, device
        self.dtype = torch.float32

    def _split(self, x):
        x = x.reshape(-1, self.dim)
        x1, x2, rest = x[:, 0], x[:, 1], x[:, 2:]
        return x1, x2 - self.b * (x1 ** 2 - self.s1 ** 2), rest

    def log_prob(self, x):
        x1, w, rest = self._split(x)
        lp = (-.5 * (x1 / self.s1) ** 2 - .5 * w ** 2 - .5 * torch.sum(rest ** 2, dim=-1)
              - .5 * self.dim * math.log(2 * math.pi) - math.log(self.s1))
        return lp.reshape(*x.shape[:-1], 1)

    def grad_log_prob(self, x):
        x1, w, rest = self._split(x)
        grad = torch.cat([(-x1 / self.s1 ** 2 + 2 * self.b * x1 * w).unsqueeze(-1),
                          -w.unsqueeze(-1), -rest], dim=-1)
        return grad.reshape(x.shape)

    def log_prob_max(self):
        """At x1 = 0, x2 = -b s1^2, x3..xd = 0."""
        mode = torch.zeros(1, self.dim, device=self.device, dtype=self.dtype)
        mode[0, 1] = -self.b * self.s1 ** 2
        return float(self.log_prob(mode))

    def sample(self, n):
        x1 = self.s1 * torch.randn(n, 1, device=self.device, dtype=self.dtype)
        x2 = torch.randn(n, 1, device=self.device, dtype=self.dtype) + self.b * (x1 ** 2 - self.s1 ** 2)
        rest = torch.randn(n, self.dim - 2, device=self.device, dtype=self.dtype)
        return torch.cat([x1, x2, rest], dim=-1)










class EllipticShell:
    """Mass concentrated on an ill-conditioned ellipsoidal shell.

        pi(x) ~ exp( -(rho(x) - r)^2 / 2 sigma^2 ),   rho(x) = sqrt(x^T Sigma^{-1} x),

    with Sigma = Q diag(lam) Q^T, lam log-spaced over [1, kappa] and Q a fixed
    random rotation. Exact draws: y = Sigma^{-1/2} x follows the spherical shell,
    whose radius has density rho^{d-1} exp(-(rho - r)^2 / 2 sigma^2) (drawn by
    inverse CDF on a grid) and whose direction is uniform.
    """

    def __init__(self, dim=10, kappa=50.0, radius=4.0, scale=0.3, rot_seed=0,
                 device='cpu', n_grid=40000):
        self.dim, self.device = dim, device
        self.dtype = torch.float32
        self.radius, self.scale = float(radius), float(scale)
        g = torch.Generator().manual_seed(int(rot_seed))
        lam = torch.logspace(0, math.log10(kappa), dim, dtype=torch.float64)
        q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g, dtype=torch.float64))
        self.half = ((q * lam.sqrt()) @ q.T).to(device)                    # Sigma^{1/2}
        self.inv_half = ((q * lam.rsqrt()) @ q.T).to(device)               # Sigma^{-1/2}
        rho = torch.linspace(1e-6, self.radius + 10 * self.scale, n_grid, dtype=torch.float64)
        logp = (dim - 1) * torch.log(rho) - (rho - self.radius) ** 2 / (2 * self.scale ** 2)
        self.rho_grid = rho.to(device)
        self.cdf = torch.cumsum(torch.softmax(logp, 0), 0).to(device)

    def _rho(self, x):
        y = x.reshape(-1, self.dim).double() @ self.inv_half
        # Clamped so the gradient stays finite at the origin
        return y, (y ** 2).sum(-1).clamp_min(1e-12).sqrt()

    def log_prob(self, x):
        _, rho = self._rho(x)
        lp = -(rho - self.radius) ** 2 / (2 * self.scale ** 2)
        return lp.reshape(*x.shape[:-1], 1).to(x.dtype)

    def grad_log_prob(self, x):
        """grad rho = Sigma^{-1} x / rho, so grad log pi = -((rho - r) / sigma^2) grad rho."""
        y, rho = self._rho(x)
        g = -((rho - self.radius) / self.scale ** 2 / rho).unsqueeze(-1) * (y @ self.inv_half)
        return g.reshape(x.shape).to(x.dtype)

    def log_prob_max(self):
        """Attained on the whole shell rho = r."""
        return 0.0

    def sample(self, n):
        u = torch.rand(n, device=self.device).double().clamp(max=1 - 1e-9)
        idx = torch.searchsorted(self.cdf.contiguous(), u.contiguous())
        r = self.rho_grid[idx.clamp(max=len(self.rho_grid) - 1)]
        d = torch.randn(n, self.dim, device=self.device).double()
        y = d / d.norm(dim=-1, keepdim=True) * r.unsqueeze(-1)
        return (y @ self.half).to(self.dtype)



TARGETS = {
    # the dimension sweep
    'banana': Banana, 'x_gmm': XGMM, 'ell_shell': EllipticShell,
    # the other targets of the paper
    'logconcave': LogConcave, 'random_gmm': RandomGMM, 'two_mode_gmm': TwoModeGMM,
    'zodmc_gmm': ZODMCGMM, 'vertex_gmm': VertexGMM,
}
