"""Forward processes U_t = a(t) X + s(t) Z with a(t)^2 + s(t)^2 = 1.

Two parametrisations of the same variance-preserving process:

- `VPSchedule`: a(t) = e^{-t}, the Ornstein-Uhlenbeck process.
- `VPLinearSchedule`: a(t) = 1 - t.

They differ only in the time variable, so they define the same process. The
VP-linear time grid is built to visit the same signal levels a as the VP one,
so the two samplers differ only by discretisation error.
"""
from math import exp, log

import torch


def _exp_linear_time_steps(n, t_max, t_min, device):
    """Geometric steps up to t = 1, uniform steps after, returned as reverse time."""
    c = 1.6 * (exp(log(t_max / t_min) / n) - 1)
    t = torch.zeros(n, device=device)
    t[0] = t_min
    exp_step = True
    for i in range(1, n):
        if exp_step:
            t[i] = t[i - 1] + c * t[i - 1]
            if t[i] >= 1:
                c = (t_max - t[i - 1]) / (n - i)
                t[i] = t[i - 1] + c
                exp_step = False
        else:
            t[i] = t[i - 1] + c
    t[-1] = t_max
    return torch.flip(t_max - t, dims=(0,))


class _VariancePreserving:
    """Everything that follows from a(t), s(t) = sqrt(1 - a^2) and their derivatives."""

    def sigma(self, t):
        """s(t)."""
        return (1 - self.scale(t) ** 2) ** 0.5

    # Reverse SDE  dx = [lam x + g2 * score] dr + sqrt(g2) dW  in reverse time r = t_max - t
    def drift_rate(self, t):
        """lam = -a'/a."""
        return -self.scale_dot(t) / self.scale(t)

    def score_coeff(self, t):
        """g2 = 2 s s' - 2 (a'/a) s^2."""
        s = self.sigma(t)
        return 2 * s * self.sigma_dot(t) - 2 * (self.scale_dot(t) / self.scale(t)) * s ** 2

    # p(x_0 | x_t = x) is proportional to pi(x_0) N(x_0; x/a, (s/a)^2 I)
    def proposal(self, x, t):
        a, s = self.scale(t), self.sigma(t)
        return x / a, s / a

    def gaussian_penalty_grad(self, x0, x, t):
        """Gradient in x_0 of log N(x; a x_0, s^2 I)."""
        a, s = self.scale(t), self.sigma(t)
        return a * (x - a * x0) / s ** 2

    def score_from_denoiser(self, m, x, t):
        """Tweedie: grad log p_t(x) = (a E[x_0 | x_t = x] - x) / s^2."""
        return (self.scale(t) * m - x) / self.sigma(t) ** 2


class VPSchedule(_VariancePreserving):
    """a(t) = e^{-t} on [t_min, t_max]."""

    def __init__(self, t_min=1e-3, t_max=5.0):
        self.t_min = t_min
        self.t_max = t_max

    def scale(self, t):
        """a(t)."""
        return torch.exp(-torch.as_tensor(t))

    def scale_dot(self, t):
        return -self.scale(t)

    def sigma_dot(self, t):
        return self.scale(t) ** 2 / self.sigma(t)

    def t_from_scale(self, a):
        return -torch.log(torch.as_tensor(a))

    @property
    def prior_std(self):
        """The stationary law is N(0, I)."""
        return 1.0

    # Closed forms of the generic expressions: lam = 1 and g2 = 2(a^2 + s^2) = 2
    def drift_rate(self, t):
        return torch.ones_like(torch.as_tensor(t))

    def score_coeff(self, t):
        return 2 * torch.ones_like(torch.as_tensor(t))

    def time_steps(self, n, device):
        """Reverse-time grid r in [0, t_max - t_min]: geometric steps near the data
        end, uniform once t passes 1."""
        return _exp_linear_time_steps(n, self.t_max, self.t_min, device)


class VPLinearSchedule(_VariancePreserving):
    """a(t) = 1 - t on [t_min, t_max]; t_max < 1, since a = 0 makes the proposal x/a singular."""

    def __init__(self, t_min=0.01, t_max=0.99):
        assert 0 < t_min < t_max < 1
        self.t_min = t_min
        self.t_max = t_max

    def scale(self, t):
        """a(t)."""
        return 1.0 - torch.as_tensor(t)

    def scale_dot(self, t):
        return -torch.ones_like(torch.as_tensor(t))

    def sigma_dot(self, t):
        return self.scale(t) / self.sigma(t)

    def t_from_scale(self, a):
        return 1.0 - torch.as_tensor(a)

    @property
    def prior_std(self):
        return float(self.sigma(torch.as_tensor(self.t_max)))

    def time_steps(self, n, device):
        """`VPSchedule`'s grid, placed at the same signal levels a and mapped to this
        time variable: dense at both ends of t and coarse in the middle. A uniform
        grid in t would put fewer steps at low a and cost raw IS about 1.7x in W2
        at 20 steps."""
        a_lo, a_hi = 1.0 - self.t_max, 1.0 - self.t_min
        u_lo, u_hi = -log(a_hi), -log(a_lo)          # the VP times with these a
        r = _exp_linear_time_steps(n, u_hi, u_lo, device)
        a = torch.exp(-(u_hi - r))
        return (self.t_max - (1.0 - a)).clamp_min(0.0)


SCHEDULES = {'vp': VPSchedule, 'vplinear': VPLinearSchedule}
