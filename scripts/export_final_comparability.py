"""Audit frozen final records and export descriptive paper evidence without simulation."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

from manipulation.act_model_selection import validate_checkpoint_binding, paired_success_counts
from manipulation.file_integrity import sha256_file
from manipulation.nominal_comparability import describe_nominal_pair
from task_registry import get_task

ROOT = Path(__file__).resolve().parents[1]
TAG = '_protocol-descriptive-20260911'


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_cell(records, summary, task, actor, rate, rate_index, checkpoint):
    """Check original final records without rewriting their factors or labels."""
    spec = get_task(task)
    require(len(records) == 200, 'cell requires 200 episodes')
    ids = [r['initial_condition_id'] for r in records]
    require(all(type(i) is int for i in ids) and set(ids) == set(range(200)), 'invalid IDs')
    required = dict(task=spec.gym_id, policy=actor, actor_spec=actor, bank_role='final',
                    seed=73001 + 1000 * rate_index, num_envs=32, jitter=0.01,
                    fixed_phase_rate_override=None)
    for key, value in required.items():
        require(all(r[key] == value for r in records), f'episode mismatch: {key}')
        require(summary[key] == value, f'summary mismatch: {key}')
    require(summary['rate'] == rate, 'summary rate mismatch')
    require(all(r['task_rate'] == rate and r['corrupt'] is False
                and r['action_noise'] == 0 for r in records), 'altered evaluation factors')
    banks = {r['initial_condition_bank_sha256'] for r in records}
    require(len(banks) == 1, 'mixed banks')
    bank = next(iter(banks))
    require(len(bank) == 64 and all(c in '0123456789abcdef' for c in bank), 'invalid bank hash')
    require(summary['episodes'] == summary['episodes_requested'] == 200, 'summary count')
    for field in ('task_success', *spec.stage_keys):
        require(all(type(r[field]) is bool for r in records), f'nonboolean {field}')
        count = sum(r[field] for r in records)
        require(summary[field]['k'] == count and summary[field]['n'] == 200, f'count {field}')
        require(abs(summary[field]['frac'] - count / 200) < 1e-12, f'fraction {field}')
    require(dict(Counter(r['failure_reason'] for r in records)) == summary['failure_reasons'],
            'failure reason count mismatch')
    validate_checkpoint_binding(records, summary, policy=actor, checkpoint_path=checkpoint)
    return bank


def paired_interval(expert, learner):
    counts = paired_success_counts(expert, learner)
    e = {r['initial_condition_id']: r for r in expert}
    a = {r['initial_condition_id']: r for r in learner}
    outcomes = np.array([int(a[i]['task_success']) - int(e[i]['task_success'])
                         for i in sorted(e)])
    means = np.random.default_rng(0).choice(outcomes, size=(20000, 200), replace=True).mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5], method='linear')
    return counts, float(lo * 100), float(hi * 100)


def write_csv(path, values):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'output already exists; choose a new export version')
    freeze_path = ROOT / 'outputs/act_comparability/final-batch-20260911/freeze.json'
    freeze = json.loads(freeze_path.read_text())
    for path, digest in freeze['sha256'].items():
        require(sha256_file(ROOT / path) == digest, f'frozen source changed: {path}')
    events = rows(freeze_path.parent / 'events.jsonl')
    completed = [e for e in events if 'exit_code' in e]
    require(len(completed) == 6 and all(e['exit_code'] == 0 for e in completed), 'batch incomplete')
    sources = {str(freeze_path.relative_to(ROOT)): sha256_file(freeze_path)}
    table, stages, nominal, expert_repeats = [], [], [], {}
    task_manifests = []
    for task in ('upright', 'peg'):
        task_manifests.append(asdict(get_task(task)))
    for run in freeze['runs']:
        task, seed = run['task'], run['seed']
        base = ROOT / run['output']
        selection_path = ROOT / run['selection']
        selection = json.loads(selection_path.read_text())
        require(sha256_file(ROOT / run['checkpoint']) == selection['selected_checkpoint_sha256'],
                'selected checkpoint changed')
        summary_path = base / f'summary{TAG}.jsonl'
        summaries = rows(summary_path)
        require(len(summaries) == 16, 'expected 16 actor/rate summaries')
        indexed = {(s['policy'], s['rate']): s for s in summaries}
        require(len(indexed) == 16, 'duplicate summary cells')
        sources[str(summary_path.relative_to(ROOT))] = sha256_file(summary_path)
        grouped = {}
        for actor in ('expert', 'act'):
            path = base / f'{actor}{TAG}.jsonl'
            records = rows(path)
            require(len(records) == 1600, 'actor record count mismatch')
            require({r['task_rate'] for r in records} == set(freeze['rates']), 'wrong rate grid')
            sources[str(path.relative_to(ROOT))] = sha256_file(path)
            for ri, rate in enumerate(freeze['rates']):
                cell = [r for r in records if r['task_rate'] == rate]
                summary = indexed[actor, rate]
                validate_cell(cell, summary, task, actor, rate, ri,
                              run['checkpoint'] if actor == 'act' else None)
                grouped[actor, rate] = cell
                for field in get_task(task).stage_keys:
                    stages.append(dict(task=task, seed=seed, actor=actor, rate=rate,
                                       stage=field, successes=summary[field]['k'], episodes=200))
        for rate in freeze['rates']:
            e, a = grouped['expert', rate], grouped['act', rate]
            counts, lo, hi = paired_interval(e, a)
            signature = [(r['initial_condition_id'], r['object_initial_pose'],
                          [r[k] for k in ('task_success', *get_task(task).stage_keys)])
                         for r in sorted(e, key=lambda r: r['initial_condition_id'])]
            key = (task, rate)
            if key in expert_repeats:
                require(signature == expert_repeats[key], 'repeated expert outcomes differ')
            else:
                expert_repeats[key] = signature
            es, ac = indexed['expert', rate], indexed['act', rate]
            table.append(dict(task=task, seed=seed, rate=rate, episodes=200,
                              expert_successes=es['task_success']['k'],
                              act_successes=ac['task_success']['k'],
                              difference_pp=counts['learner_minus_expert'] * 100,
                              ci_lower_pp=lo, ci_upper_pp=hi,
                              both=counts['both_succeed'], expert_only=counts['expert_only_succeeds'],
                              act_only=counts['learner_only_succeeds'], neither=counts['neither_succeeds'],
                              demonstrated_lo=get_task(task).demonstrated_range[0],
                              demonstrated_hi=get_task(task).demonstrated_range[1]))
            if rate == 1:
                nominal.append(dict(task_alias=task, seed=seed, **describe_nominal_pair(e, a)))
    args.output.mkdir(parents=True)
    write_csv(args.output / 'paired_results.csv', table)
    write_csv(args.output / 'stage_counts.csv', stages)
    payload = dict(status='validated final evaluation records', episodes=19200,
                   independent_expert_episodes=3200, act_episodes=9600,
                   expert_repeated_across_seeds=True, repeated_expert_outcomes_identical=True,
                   cells=96, paired_comparisons=48,
                   interval='pointwise two-sided 95% paired percentile bootstrap; 20000 resamples; seed 0; linear percentiles',
                   limitations=['No equivalence or noninferiority decision.',
                                'Rates use different banks; do not treat cross-rate differences as paired.',
                                'Identical paired outcomes yield degenerate bootstrap intervals.',
                                'Source hashes checked retrospectively against the pre-run freeze; assets were not included in that freeze.',
                                'No new hardware or cross-embodiment success is established.'],
                   source_sha256=sources, exporter_sha256=sha256_file(Path(__file__)),
                   command=' '.join(sys.argv), numpy_version=np.__version__,
                   results=table, nominal=nominal, task_manifests=task_manifests)
    (args.output / 'evidence.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({k: v for k, v in payload.items() if k in ('status','episodes','cells','paired_comparisons')}))
    for r in table:
        if r['rate'] in (1, 2):
            print(r)


if __name__ == '__main__':
    main()
