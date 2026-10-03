"""Affine post-processing of a base score estimator (our method).

At a query (x, b*), the base estimator is evaluated at design levels
b_0 = b*, b_1, ..., b_{n-1}, and the exact scores at the endpoints b = 0
(the prior, -x) and b = 1 (the target, grad log pi) are appended. The output
is h^T s, where the weights h solve

    h = argmin_h  h^T Sigma h + mu * sum_k w_k (A h - phi(b*))_k^2,

with A_{ki} = phi_k(b_i) the Legendre basis at the levels, Sigma the covariance
of the base estimates, w_k = 1/k^3 (w_0 large, so the constant is preserved
almost exactly) and mu = lam * tr(Sigma)/n. The solution is

    h = mu (Sigma + mu A^T W A)^{-1} A^T W phi(b*).

Sigma is estimated at the query from `groups` independent replicates, and the
replicate mean is the estimate the weights are applied to.

Two independent choices set how many weight vectors there are:

- `sigma`: what Sigma is pooled over across the batch. 'global'
  pools every particle into one Sigma; 'point' estimates a Sigma for
  each particle from its own replicates.
- `weights`: 'estimate' applies one h to the whole score vector, so Sigma is
  also pooled over the d coordinates; 'coordinate' gives each coordinate j its
  own Sigma_j, mu_j and h_j, so coordinates whose estimates have different noise
  profiles across levels are weighted differently.

Sigma has (groups - 1) x (particles pooled) x (coordinates pooled) degrees of
freedom, and the solve needs at least `num_levels` of them: 'point' together
with 'coordinate' therefore needs groups >= num_levels + 1.
"""
import torch


def legendre_basis(b, K):
    """(N, K+1) matrix of Legendre polynomials L_k(2b - 1), k = 0..K."""
    z = 2.0 * b - 1.0
    P = torch.empty(z.shape[0], K + 1, dtype=z.dtype, device=z.device)
    P[:, 0] = 1.0
    if K >= 1:
        P[:, 1] = z
    for k in range(1, K):
        P[:, k + 1] = ((2 * k + 1) * z * P[:, k] - k * P[:, k - 1]) / (k + 1)
    return P


class PostProcessed:
    def __init__(self, base, degree=5, num_levels=6, groups=2, lam=1.0,
                 b_min=0.05, b_max=0.95, w0=1e4, endpoint_var=1e-4, sigma_ridge=1e-8,
                 sigma='global', weights='estimate'):
        assert sigma in ('global', 'point') and weights in ('estimate', 'coordinate')
        self.base, self.sigma, self.weights = base, sigma, weights
        self.target, self.schedule = base.target, base.schedule
        self.integrator = base.integrator
        self.K, self.num_levels, self.groups, self.lam = degree, num_levels, groups, lam
        self.b_min, self.b_max = b_min, b_max
        self.endpoint_var, self.sigma_ridge = endpoint_var, sigma_ridge
        k = torch.arange(degree + 1, dtype=torch.float64)
        self.w = torch.where(k >= 1, k.clamp_min(1) ** -3.0, torch.full_like(k, w0))

    @property
    def calls_per_query(self):
        """Base-estimator calls per score query."""
        return self.groups * self.num_levels

    def levels(self, b_star, device, dtype):
        """A uniform grid on [b_min, b_max] with its point nearest b* replaced by b*."""
        grid = torch.linspace(self.b_min, self.b_max, self.num_levels, device=device, dtype=dtype)
        drop = int(torch.argmin((grid - b_star).abs()))
        keep = torch.cat([grid[:drop], grid[drop + 1:]])
        return torch.cat([torch.as_tensor([b_star], device=device, dtype=dtype), keep])

    @torch.no_grad()
    def score(self, x, t):
        device, dtype = x.device, x.dtype
        b_star = float(self.schedule.scale(torch.as_tensor(t, dtype=dtype, device=device)))
        levels = self.levels(b_star, device, dtype)
        times = [self.schedule.t_from_scale(b) for b in levels]
        M, d = x.shape
        n = len(levels)

        # Base estimates: S[g, i] is replicate g at level b_i
        S = torch.stack([torch.stack([self.base.score(x, t_i) for t_i in times])
                         for _ in range(self.groups)])                      # (G, n, M, d)
        s = S.mean(dim=0)                                                   # (n, M, d)

        # Particles where the base estimator diverged (RDMC chains occasionally
        # do) are left out of Sigma, and their output stays non-finite
        finite = (torch.isfinite(S).all(dim=0).all(dim=0).all(dim=-1)
                  & torch.isfinite(s).all(dim=0).all(dim=-1))               # (M,)
        n_bad = int((~finite).sum())
        if n_bad == M:
            return s[0]

        # Sigma from the replicate deviations, pooled over the chosen axes: (Md, Dd, n, n)
        # with Md in {1, M} and Dd in {1, d}
        pool_m, pool_d = self.sigma == 'global', self.weights == 'estimate'
        m_fin = int(finite.sum())
        per_group = (m_fin if pool_m else 1) * (d if pool_d else 1)
        dof = (self.groups - 1) * per_group
        if dof < n:
            raise ValueError(f'Sigma has {dof} degrees of freedom for {n} levels (groups='
                             f'{self.groups}, sigma={self.sigma!r}, weights={self.weights!r}); '
                             f'needs groups >= {-(-n // per_group) + 1}')
        dev = S - S.mean(dim=0, keepdim=True)
        if n_bad:
            dev = torch.nan_to_num(dev, nan=0.0, posinf=0.0, neginf=0.0) * finite.view(1, 1, -1, 1)
        dev = dev.double()
        kept = ('' if pool_m else 'm') + ('' if pool_d else 'd')
        Sigma = torch.einsum(f'gimd,gjmd->{kept}ij', dev, dev) / dof
        Sigma = Sigma.reshape(1 if pool_m else M, 1 if pool_d else d, n, n)
        eye = torch.eye(n, device=device, dtype=Sigma.dtype)
        if n_bad and not pool_m:   # their Sigma is exactly 0; keep the solve well-posed
            Sigma[~finite] = eye

        # Where the base estimator returned the same value in every replicate, Sigma = 0
        # and there is nothing to weigh: those outputs are the on-policy estimate
        est_scale = float((s[:, finite, :].double() ** 2).mean()) if n_bad else float((s ** 2).mean())
        degenerate = torch.diagonal(Sigma, dim1=-2, dim2=-1).mean(-1) <= 1e-12 * max(est_scale, 1e-30)
        degenerate = degenerate.expand(M, d)                                 # (M, d)
        if bool(degenerate.all()):
            return s[0]

        A = legendre_basis(levels, self.K).T                                # (K+1, n)
        phi = legendre_basis(torch.as_tensor([b_star], device=device, dtype=dtype), self.K).squeeze(0)

        # Exact endpoints: prior score at b = 0, target score at b = 1. They get a small
        # variance, relative to their own magnitude, instead of being hard constraints
        ends = torch.as_tensor([0.0, 1.0], device=device, dtype=dtype)
        A = torch.cat([A, legendre_basis(ends, self.K).T], dim=1)           # (K+1, n+2)
        s_end = torch.stack([-x, self.target.grad_log_prob(x)])             # (2, M, d)
        s = torch.cat([s, s_end], dim=0)
        s_end_fin = s_end[:, finite, :] if (n_bad and pool_m) else s_end
        se2 = torch.nan_to_num(s_end_fin.double() ** 2, nan=0.0, posinf=0.0, neginf=0.0)
        dims = tuple(ax for ax, pool in ((1, pool_m), (2, pool_d)) if pool)
        ev = self.endpoint_var * (se2.mean(dim=dims, keepdim=True) if dims else se2)   # (2, Md, Dd)
        Sigma_aug = torch.zeros(*Sigma.shape[:-2], n + 2, n + 2, device=device, dtype=Sigma.dtype)
        Sigma_aug[..., :n, :n] = Sigma
        Sigma_aug[..., n, n], Sigma_aug[..., n + 1, n + 1] = ev[0], ev[1]

        h = self.solve(Sigma_aug, A, phi, n).to(dtype)                       # (Md, Dd, n+2)
        if h.shape[:2] == (1, 1):
            out = torch.einsum('n,nmd->md', h[0, 0], s)
        else:
            out = torch.einsum('mdn,nmd->md', h.expand(M, d, n + 2), s)
        if bool(degenerate.any()):
            out = torch.where(degenerate, s[0], out)
        if n_bad:
            out = out.masked_fill(~finite.view(-1, 1), float('nan'))
        return out

    def solve(self, Sigma, A, phi, n):
        """h = mu (Sigma + mu A^T W A)^{-1} A^T W phi, with mu = lam * mean of the first n
        diagonal entries of Sigma (the base estimator's own, not the endpoints').
        Sigma may carry leading batch dimensions, one solve per entry."""
        Sig = Sigma + self.sigma_ridge * torch.eye(Sigma.shape[-1], device=Sigma.device, dtype=Sigma.dtype)
        A, phi = A.to(Sig.dtype), phi.to(Sig.dtype)
        mu = (self.lam * torch.diagonal(Sig, dim1=-2, dim2=-1)[..., :n].mean(-1))[..., None, None]
        AtW = A.T * self.w.to(device=Sig.device, dtype=Sig.dtype)
        lhs = Sig + mu * (AtW @ A)
        rhs = (mu.squeeze(-1) * (AtW @ phi)).unsqueeze(-1)
        return torch.linalg.solve(lhs, rhs).squeeze(-1)
