"""Summarize paired NKSID runs and plot the measured OSCR values."""
import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs'


def load(name):
    folder = OUTPUT / name
    config = json.loads((folder / 'config.json').read_text())
    result = json.loads((folder / 'results.json').read_text())
    assert set(result['variants']) == {item['name'] for item in config['variants']}
    assert all(len(v['loss_history']) == config['epochs'] for v in result['variants'].values())
    return result['variants'], (folder / 'split.json').read_bytes()


def main():
    groups = {}
    expected_split = None
    for group in ('sweep', 'sapfs_controls'):
        first, split_a = load(f'nksid_{group}')
        second, split_b = load(f'nksid_{group}_seed123')
        expected_split = expected_split or split_a
        assert split_a == split_b == expected_split
        assert first.keys() == second.keys()
        groups[group] = first, second
    out = OUTPUT / 'nksid_sensitivity_report'
    out.mkdir(exist_ok=True)
    keys = ('known_top1', 'known_correct_accepted', 'unknown_recall', 'oscr_macro')
    with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['group', 'variant', 'tail_classes'] +
                        [f'{key}_{suffix}_pct' for key in keys for suffix in ('42', '123', 'mean')])
        for group, (a, b) in groups.items():
            for name in a:
                assert a[name]['tail_classes'] == b[name]['tail_classes']
                fields = []
                for key in keys:
                    pair = [100 * run[name][key] for run in (a, b)]
                    fields.extend(round(v, 4) for v in (*pair, mean(pair)))
                writer.writerow([group, name, '+'.join(a[name]['tail_classes']), *fields])
    a, b = groups['sweep']
    categories = [
        ('SAPFS retention bounds', [n for n in a if n.startswith('bounds_')] + ['reference'], None),
        ('Prototype EMA momentum', ['momentum_0', 'momentum_0.5', 'momentum_0.7', 'reference', 'momentum_0.99'], ['0', '.5', '.7', '.9', '.99']),
        ('Contrastive loss weight', ['weight_0.5', 'weight_1', 'weight_2', 'reference', 'weight_4'], ['.5', '1', '2', '3', '4']),
        ('Tail-frequency quantile', ['rho_0.1', 'reference', 'rho_0.3', 'rho_0.4', 'rho_0.5'], ['.1', '.2', '.3', '.4', '.5']),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for axis, (title, names, labels) in zip(axes.flat, categories):
        if labels is None:
            labels = [n.removeprefix('bounds_').replace('_', ', ') if n != 'reference' else '0.2, 0.8' for n in names]
        x = range(len(names))
        curves = [[100 * run[n]['oscr_macro'] for n in names] for run in (a, b)]
        for label, values in zip(('Seed 42', 'Seed 123'), curves):
            axis.plot(x, values, 'o', alpha=.65, label=label, linestyle='none' if title.startswith('SAPFS') else '--')
        axis.plot(x, list(map(mean, zip(*curves))), 'D', color='black', label='Mean')
        axis.axhline(mean([100 * run['baseline']['oscr_macro'] for run in (a, b)]), color='gray', linestyle=':', label='Baseline mean')
        axis.set(title=title, ylabel='Macro OSCR (%)', xticks=list(x), xticklabels=labels)
        axis.tick_params(axis='x', labelrotation=25 if title.startswith('SAPFS') else 0)
        axis.grid(axis='y', alpha=.25)
    axes[0, 0].legend(fontsize=8, ncols=2)
    fig.savefig(out / 'sensitivity.png', dpi=180)


if __name__ == '__main__':
    main()

