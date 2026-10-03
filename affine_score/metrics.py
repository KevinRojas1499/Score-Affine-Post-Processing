import ot
import torch


def w2(x, y):
    """Wasserstein-2 distance between two empirical measures; nan if it cannot be trusted."""
    M = ot.dist(x, y)
    # A diverged sampler can return finite samples large enough that the squared
    # distances overflow, and the solver then returns 0 instead of failing
    if not torch.isfinite(M).all():
        return float('nan')
    a = torch.ones(len(x), device=x.device) / len(x)
    b = torch.ones(len(y), device=y.device) / len(y)
    out = float(ot.emd2(a, b, M)) ** 0.5
    return float('nan') if out == 0.0 and not torch.equal(x, y) else out


def w2_floor(target, reference, num_samples, seed=0, reps=5):
    """W2 of exact draws of `num_samples` points: the best any correct sampler can score.

    W2 between finite sample sets does not go to zero, and in high dimension
    this floor is most of the number, so every result is read against it.
    """
    vals = []
    for i in range(reps):
        torch.manual_seed(10000 + 97 * seed + i)
        vals.append(w2(reference, target.sample(num_samples).to(reference)))
    return sum(vals) / reps
