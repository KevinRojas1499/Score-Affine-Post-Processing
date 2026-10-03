"""W2 against dimension for IS and RDMC, with and without our post-processing, and ZOD-MC.

    uv run python3 dimension_sweep.py                          # the full sweep
    uv run python3 dimension_sweep.py --targets banana --dims 2,10 --seeds 0

Every method gets the same number of target-oracle calls per reverse step
(`--budget`). The post-processor makes groups x 6 base-estimator calls per step
(12 by default), so its base estimator gets budget / 12 per call. Results are cached per
cell and setting in `--out`, so an interrupted run resumes where it
stopped and a finished cell is never recomputed (pass `--force` to redo). Plot
with `plot_dimension_sweep.py`.
"""
import argparse
import os
import time

import torch

from affine_score import (SCHEDULES, TARGETS, ImportanceSampling, PostProcessed, RDMC,
                          ZODMC, sample, w2, w2_floor)

# The paper's grid. Any other target in affine_score.TARGETS runs with --dims (or at
# its own dimension if it has no dimension knob); nothing below is tuned for those.
DIMS = {'banana': (2, 5, 10, 20, 30, 50),
        'x_gmm': (2, 4, 6, 8, 12, 16),
        'ell_shell': (2, 5, 10, 20, 30, 50)}
# Constant RDMC inner step, the best of a sweep of raw RDMC's own W2 per target
RDMC_STEP = {'banana': 2e-4, 'x_gmm': 7e-3, 'ell_shell': 7e-3}
RDMC_STEP_DEFAULT = 1e-3
# Samples farther than this from the origin are counted as diverged
MAX_ABS = {'x_gmm': 1e4}
METHODS = ('is', 'is_ours', 'rdmc', 'rdmc_ours', 'zodmc')


def make_estimator(method, target, schedule, budget, rdmc_step, rdmc_steps, groups=2,
                   sigma='global', weights='estimate'):
    ours = method.endswith('_ours')
    # oracle calls per base-estimator call; ours makes groups x 6 levels calls per step
    calls = budget // (groups * 6) if ours else budget
    if method.startswith('is'):
        base = ImportanceSampling(target, schedule, num_samples=500, num_batches=calls // 500)
    elif method.startswith('zodmc'):
        base = ZODMC(target, schedule, num_samples=500, num_batches=calls // 500)
    else:
        base = RDMC(target, schedule, num_chains=calls // rdmc_steps, steps=rdmc_steps,
                    step_size=rdmc_step)
    return PostProcessed(base, groups=groups, sigma=sigma, weights=weights) if ours else base


def run_cell(name, dim, seed, methods, args, results, device):
    target = TARGETS[name](dim=dim, device=device)
    schedule = SCHEDULES[args.schedule]()
    torch.manual_seed(seed)
    reference = target.sample(args.num_reference)

    key = (name, dim, 'floor', seed, args.num_samples)
    if args.force or key not in results:
        results[key] = dict(w2=w2_floor(target, reference, args.num_samples, seed=seed))
        save(results, args.out)
    print(f'{name:<10}{dim:>4}{seed:>5}  {"floor":<10}{results[key]["w2"]:>10.3f}')

    for method in methods:
        key = (name, dim, method, seed, args.num_samples, args.budget, args.disc_steps)
        key += settings_suffix(method, args.schedule, args.groups, args.sigma, args.weights)
        if args.force or key not in results:
            est = make_estimator(method, target, schedule, args.budget,
                                 RDMC_STEP.get(name, RDMC_STEP_DEFAULT),
                                 args.rdmc_steps, args.groups, args.sigma, args.weights)
            start = time.time()
            torch.manual_seed(seed)
            try:
                x = sample(est, args.num_samples, args.disc_steps, device)
            except ValueError as e:
                # Sigma underdetermined at this (groups, sigma, weights, dim): the
                # setting is not defined here, so the cell is recorded as such
                results[key] = dict(w2=float('nan'), nan_frac=float('nan'), disp=float('nan'),
                                    empty_frac=None, seconds=0.0, undefined=str(e))
                save(results, args.out)
                print(f'{name:<10}{dim:>4}{seed:>5}  {method:<10}{"undefined":>10}   {e}')
                continue
            bad = ~torch.isfinite(x).all(dim=1)
            if name in MAX_ABS:
                bad |= x.abs().amax(dim=1) > MAX_ABS[name]
            x = x[~bad]
            base = getattr(est, 'base', est)
            results[key] = dict(w2=w2(reference, x) if len(x) >= 2 else float('nan'),
                                nan_frac=float(bad.float().mean()),
                                # Below 1 means under-dispersed, which can score under the W2 floor
                                disp=float(torch.cov(x.T).trace() / torch.cov(reference.T).trace()),
                                # ZOD-MC only: queries where no proposal was accepted
                                empty_frac=getattr(base, 'empty_frac', None),
                                seconds=time.time() - start, samples=x.float().cpu())
            save(results, args.out)
        r = results[key]
        empty = '' if r.get('empty_frac') is None else f'   empty_frac={r["empty_frac"]:.3f}'
        disp = f'   disp={r["disp"]:.2f}' if 'disp' in r else ''
        print(f'{name:<10}{dim:>4}{seed:>5}  {method:<10}{r["w2"]:>10.3f}'
              f'   nan_frac={r["nan_frac"]:.3f}{disp}{empty}   {r["seconds"]:.0f}s')


def settings_suffix(method, schedule='vp', groups=2, sigma='global', weights='estimate'):
    """Cache-key entries for the non-default settings that affect this method."""
    suffix = () if schedule == 'vp' else (schedule,)
    if method.endswith('_ours'):
        suffix += tuple(f'{k}={v}' for k, v, default in (('groups', groups, 2), ('sigma', sigma, 'global'),
                                                        ('weights', weights, 'estimate')) if v != default)
    return suffix


def load(path):
    return torch.load(path, weights_only=False) if os.path.exists(path) else {}


def save(results, path):
    """Write to a temporary file and rename, so a crash never corrupts the cache."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(results, path + '.tmp')
    os.replace(path + '.tmp', path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--targets', default=','.join(DIMS), help='comma list from ' + ', '.join(TARGETS))
    p.add_argument('--dims', default=None, help='comma list; default: the paper grid per target')
    p.add_argument('--methods', default=','.join(METHODS))
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--budget', type=int, default=60000, help='oracle calls per reverse step')
    p.add_argument('--disc-steps', type=int, default=20)
    p.add_argument('--num-samples', type=int, default=500)
    p.add_argument('--num-reference', type=int, default=2000, help='exact draws W2 is measured against')
    p.add_argument('--schedule', default='vp', choices=sorted(SCHEDULES),
                   help='vp: a = exp(-t); vplinear: a = 1 - t. The same process in two time variables')
    p.add_argument('--groups', type=int, default=2, help='ours only: replicate groups Sigma is estimated from')
    p.add_argument('--sigma', default='global', choices=('global', 'point'),
                   help='ours only: one Sigma for the batch, or one per particle')
    p.add_argument('--weights', default='estimate', choices=('estimate', 'coordinate'),
                   help='ours only: one h per score estimate, or one per coordinate')
    p.add_argument('--rdmc-steps', type=int, default=1000, help='length of each RDMC inner chain')
    p.add_argument('--out', default='results/dimension_sweep.pt')
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    methods = args.methods.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]
    results = load(args.out)
    print(f'{"target":<10}{"dim":>4}{"seed":>5}  {"method":<10}{"W2":>10}')
    for name in args.targets.split(','):
        if args.dims:
            dims = [int(d) for d in args.dims.split(',')]
        else:
            dims = DIMS.get(name) or (TARGETS[name](device=device).dim,)
        for dim in dims:
            for seed in seeds:
                run_cell(name, dim, seed, methods, args, results, device)


if __name__ == '__main__':
    main()
