"""Plot the cached dimension sweep and print it as a table.

    uv run python3 plot_dimension_sweep.py

Reads `--results` (written by dimension_sweep.py) and writes `--out`. One panel
per target: W2 against dimension, the mean over seeds with a [min, max] bar,
and the W2 of exact draws (the floor) in grey.
"""
import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from dimension_sweep import settings_suffix

LABELS = {'is': 'IS', 'is_ours': 'IS + ours', 'rdmc': 'RDMC', 'rdmc_ours': 'RDMC + ours',
          'zodmc': 'ZOD-MC', 'zodmc_ours': 'ZOD-MC + ours'}
COLORS = {'is': '#4C72B0', 'is_ours': '#DD8452', 'rdmc': '#55A868', 'rdmc_ours': '#8172B3',
          'zodmc': '#C44E52', 'zodmc_ours': '#937860'}
TITLES = {'banana': 'Banana', 'x_gmm': 'X-GMM', 'ell_shell': r'Ellipsoidal shell ($\kappa=50$)'}


def collect(results, budget, disc_steps, num_samples, metric='w2', **settings):
    """{target: {method: {dim: [metric per seed]}}}, with the floor under method 'floor'."""
    table = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for key, r in sorted(results.items(), key=lambda kv: kv[0][3]):
        name, dim, method, seed, n = key[:5]
        if n != num_samples or (method != 'floor' and key[5:] != (budget, disc_steps)
                                + settings_suffix(method, **settings)):
            continue
        if metric in r:
            table[name][method][dim].append(r[metric])
        elif method == 'floor':           # exact draws are dispersed exactly right
            table[name][method][dim].append(1.0)
    return table


def print_table(table, title='W2 (W2 / floor)', ratio=True):
    for name, rows in table.items():
        dims = sorted(rows['floor'])
        print(f'\n{name}: {title}, mean over seeds')
        print(f'{"dim":>14}' + ''.join(f'{d:>16}' for d in dims))
        for method in ['floor'] + [m for m in LABELS if m in rows]:
            cells = []
            for d in dims:
                vals = rows[method].get(d)
                if not vals:
                    cells.append(f'{"-":>16}')
                    continue
                m = np.mean(vals)
                r = '' if method == 'floor' or not ratio else f' ({m / np.mean(rows["floor"][d]):.2f}x)'
                cells.append(f'{m:.2f}{r}'.rjust(16))
            print(f'{LABELS.get(method, method):>14}' + ''.join(cells))


def plot(table, out):
    names = [n for n in TITLES if n in table]
    fig, axes = plt.subplots(1, len(names), figsize=(4.8 * len(names), 3.7), squeeze=False)
    for i, (ax, name) in enumerate(zip(axes[0], names)):
        rows = table[name]
        dims = np.array(sorted(rows['floor']))
        ax.plot(dims, [np.mean(rows['floor'][d]) for d in dims], color='0.5', ls='--', lw=1.5,
                label='exact draws (floor)')
        for method in LABELS:
            ds = [d for d in dims if rows[method].get(d)]
            if not ds:
                continue
            vals = [np.array(rows[method][d], dtype=float) for d in ds]
            mean = np.array([v.mean() for v in vals])
            lo = mean - np.array([v.min() for v in vals])
            hi = np.array([v.max() for v in vals]) - mean
            ax.errorbar(ds, mean, yerr=[lo, hi], marker='o', ms=5, capsize=3, lw=2.0,
                        color=COLORS[method], label=LABELS[method])
        ax.set_title(f'({chr(ord("a") + i)}) {TITLES[name]}')
        ax.set_yscale('log')
        ax.set_xlabel('dimension $d$')
        if i == 0:
            ax.set_ylabel('$W_2$')
        ax.grid(alpha=0.3, which='both')
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False, fontsize=11)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    print(f'\nwrote {out}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results/dimension_sweep.pt')
    p.add_argument('--out', default=None, help='default: figures/dimension_sweep[_<non-default settings>].pdf')
    p.add_argument('--budget', type=int, default=60000)
    p.add_argument('--disc-steps', type=int, default=20)
    p.add_argument('--num-samples', type=int, default=500)
    p.add_argument('--schedule', default='vp', choices=('vp', 'vplinear'))
    p.add_argument('--groups', type=int, default=2)
    p.add_argument('--sigma', default='global', choices=('global', 'point'))
    p.add_argument('--weights', default='estimate', choices=('estimate', 'coordinate'))
    args = p.parse_args()
    results = torch.load(args.results, weights_only=False)
    settings = dict(schedule=args.schedule, groups=args.groups, sigma=args.sigma, weights=args.weights)
    table = collect(results, args.budget, args.disc_steps, args.num_samples, **settings)
    print_table(table)
    # W2 against a finite reference rewards under-dispersed samples: a sampler that
    # collapses towards the mean can score below the floor. tr Cov(samples) /
    # tr Cov(target) shows it.
    print_table(collect(results, args.budget, args.disc_steps, args.num_samples, 'disp', **settings),
                title='dispersion tr Cov(samples) / tr Cov(target)', ratio=False)
    tag = ''.join('_' + x.replace('=', '') for x in settings_suffix('_ours', **settings))
    plot(table, args.out or f'figures/dimension_sweep{tag}.pdf')


if __name__ == '__main__':
    main()
