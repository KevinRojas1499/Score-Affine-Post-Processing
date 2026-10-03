"""Monte Carlo score estimators and the reverse diffusion that uses them.

Both estimators approximate the denoiser E[x_0 | x_t = x] under
p(x_0 | x_t = x), which is proportional to pi(x_0) N(x_0; x/a, (s/a)^2 I),
and turn it into a score with Tweedie's formula.
"""
import torch


class ImportanceSampling:
    """Self-normalised importance sampling with the Gaussian factor as proposal.

    Each call spends num_samples * num_batches target-oracle calls per particle.
    """
    integrator = 'euler'

    def __init__(self, target, schedule, num_samples, num_batches=1):
        self.target, self.schedule = target, schedule
        self.num_samples, self.num_batches = num_samples, num_batches

    def denoiser(self, x, t):
        center, std = self.schedule.proposal(x, t)
        big_x = center.repeat_interleave(self.num_samples, dim=0)
        n, d, S = x.shape[0], x.shape[-1], self.num_samples
        # Weights are accumulated over batches with a running max, so exp never overflows
        run_max = torch.full((n, 1), float('-inf'), device=x.device, dtype=x.dtype)
        num = torch.zeros((n, d), device=x.device, dtype=x.dtype)
        den = torch.zeros((n, 1), device=x.device, dtype=x.dtype)
        for _ in range(self.num_batches):
            proposals = big_x + torch.randn(big_x.shape, device=x.device, dtype=x.dtype) * std
            logw = self.target.log_prob(proposals).view(n, S, 1)
            new_max = torch.maximum(run_max, logw.amax(dim=1))
            # Where every weight underflowed, any finite offset gives the same zero weights
            off = torch.where(torch.isfinite(new_max), new_max, torch.zeros_like(new_max))
            w = torch.exp(logw - off.unsqueeze(1))
            rescale = torch.exp(run_max - off)
            num = num * rescale + (w * proposals.view(n, S, d)).sum(dim=1)
            den = den * rescale + w.sum(dim=1)
            run_max = new_max
        # If no proposal carried any weight, fall back to the proposal centre
        return torch.where(den > 0, num / den.clamp_min(torch.finfo(den.dtype).tiny), center)

    def score(self, x, t):
        return self.schedule.score_from_denoiser(self.denoiser(x, t), x, t)


class RDMC:
    """Unadjusted Langevin chains on p(x_0 | x_t = x), started from the Gaussian factor.

    Each call spends num_chains * steps target-gradient calls per particle.
    """
    integrator = 'exponential'

    def __init__(self, target, schedule, num_chains, steps, step_size):
        self.target, self.schedule = target, schedule
        self.num_chains, self.steps, self.step_size = num_chains, steps, step_size

    def denoiser(self, x, t):
        center, std = self.schedule.proposal(x, t)
        big_x = x.repeat_interleave(self.num_chains, dim=0)
        big_center = center.repeat_interleave(self.num_chains, dim=0)
        y = big_center + torch.randn_like(big_x) * std
        h = self.step_size
        for _ in range(self.steps):
            grad = (self.target.grad_log_prob(y)
                    + self.schedule.gaussian_penalty_grad(y, big_x, t))
            y = y + torch.nan_to_num(grad) * h + (2 * h) ** .5 * torch.randn_like(y)
        return y.view(-1, self.num_chains, x.shape[-1]).mean(dim=1)

    def score(self, x, t):
        return self.schedule.score_from_denoiser(self.denoiser(x, t), x, t)


class ZODMC:
    """Rejection sampling of p(x_0 | x_t = x) (ZOD-MC), with the Gaussian factor as proposal.

    A proposal y is accepted with probability pi(y) / max pi, so the accepted draws
    are exact samples of the posterior. Each call spends num_samples * num_batches
    target-oracle calls per particle, the same as importance sampling.

    When no proposal is accepted for a particle the denoiser returns 0, as the
    reference implementation does, which pulls that particle towards the origin.
    `empty_frac` counts how often that happened.
    """
    integrator = 'euler'

    def __init__(self, target, schedule, num_samples, num_batches=1):
        self.target, self.schedule = target, schedule
        self.num_samples, self.num_batches = num_samples, num_batches
        self.log_max = float(target.log_prob_max())
        self.queries, self.empty = 0, 0

    @property
    def empty_frac(self):
        return self.empty / max(self.queries, 1)

    def denoiser(self, x, t):
        center, std = self.schedule.proposal(x, t)
        big_x = center.repeat_interleave(self.num_samples, dim=0)
        n, d, S = x.shape[0], x.shape[-1], self.num_samples
        total = torch.zeros((n, d), device=x.device, dtype=x.dtype)
        accepted = torch.zeros((n, 1), device=x.device, dtype=x.dtype)
        for _ in range(self.num_batches):
            proposals = big_x + torch.randn(big_x.shape, device=x.device, dtype=x.dtype) * std
            logp = self.target.log_prob(proposals).view(n * S, 1)
            # In log space, so a target whose maximum density underflows still works
            acc = torch.log(torch.rand((n * S, 1), device=x.device)) <= logp - self.log_max
            total += (proposals * acc).view(n, S, d).sum(dim=1)
            accepted += acc.view(n, S).sum(dim=1, keepdim=True)
        self.queries += n
        self.empty += int((accepted == 0).sum())
        return total / accepted.clamp_min(1)

    def score(self, x, t):
        return self.schedule.score_from_denoiser(self.denoiser(x, t), x, t)


@torch.no_grad()
def sample(estimator, num_samples, disc_steps, device):
    """Integrate the reverse SDE  dx = [lam x + g2 * score] dr + sqrt(g2) dW  from the prior.

    'euler' is Euler-Maruyama; 'exponential' integrates the linear drift exactly
    and holds only the score constant over each step.
    """
    schedule, target = estimator.schedule, estimator.target
    x = schedule.prior_std * torch.randn((num_samples, target.dim), device=device, dtype=target.dtype)
    r_grid = schedule.time_steps(disc_steps, device)
    for i in range(len(r_grid) - 1):
        dr = r_grid[i + 1] - r_grid[i]
        t = schedule.t_max - r_grid[i]
        score = estimator.score(x, t)
        lam, g2 = schedule.drift_rate(t), schedule.score_coeff(t)
        if estimator.integrator == 'exponential':
            e_h = torch.exp(lam * dr)
            x = (e_h * x + (e_h - 1) / lam * g2 * score
                 + ((e_h ** 2 - 1) / (2 * lam) * g2) ** .5 * torch.randn_like(x))
        else:
            x = (x + (lam * x + g2 * score) * dr
                 + g2 ** 0.5 * torch.randn(x.shape, device=device, dtype=x.dtype) * torch.abs(dr) ** 0.5)
    return x
