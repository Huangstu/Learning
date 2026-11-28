"""Compare nested NKSID training sets at matched and extended update budgets."""
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1] / 'outputs'
METRICS = ('oscr_macro', 'known_top1', 'known_correct_accepted', 'unknown_recall')


def read(run, name):
    return json.loads((ROOT / run / name).read_text())


def main():
    small_runs = ['nksid_sweep', 'nksid_sweep_seed123']
    large_runs = ['nksid_large_42', 'nksid_large_123']
    results = {}
    for seed, small_dir, large_dir in zip((42, 123), small_runs, large_runs):
        small_split = read(small_dir, 'split.json')
        large_split = read(large_dir, 'split.json')
        for key in ('validation', 'test'):
            assert small_split[key] == large_split[key], key
        assert {tuple(row) for row in small_split['train']} < {
            tuple(row) for row in large_split['train']}
        small = read(small_dir, 'results.json')['variants']
        large = read(large_dir, 'results.json')['variants']
        config = read(large_dir, 'config.json')
        assert set(large) == {variant['name'] for variant in config['variants']}
        steps_small = len(small_split['train']) // config['batch'] * 10
        steps_large = len(large_split['train']) // config['batch'] * config['epochs']
        for name, variant in large.items():
            assert variant['parameters'] == small[name]['parameters']
            assert variant['optimizer_steps'] == steps_large
            for protocol, steps, values in [
                ('small_matched', steps_small, small[name]),
                ('large_matched', steps_small, variant['intermediate'][str(steps_small)]),
                ('large_extended', steps_large, variant),
            ]:
                results.setdefault((name, protocol, steps), {})[seed] = {
                    metric: 100 * values[metric] for metric in METRICS}
    out = ROOT / 'nksid_scale_report'
    out.mkdir(exist_ok=True)
    with (out / 'comparison.csv').open('w', newline='', encoding='utf8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['variant', 'protocol', 'updates'] + [
            f'{metric}_{suffix}_pct' for metric in METRICS for suffix in ('42', '123', 'mean')])
        for (name, protocol, steps), pair in results.items():
            values = []
            for metric in METRICS:
                a, b = pair[42][metric], pair[123][metric]
                values.extend(round(v, 4) for v in (a, b, mean((a, b))))
            writer.writerow([name, protocol, steps, *values])
            print(f'{name:18} {protocol:15} {steps:4} OSCR {values[0]:5.2f} / {values[1]:5.2f}, mean {values[2]:5.2f}')


if __name__ == '__main__':
    main()
