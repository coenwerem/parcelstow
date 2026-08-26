# Reproducing the paper

The release contract is numerical. The released episode records plus the
analysis code in this repository reproduce the quantitative results the
paper reports. Principal evaluation plots can also be regenerated from
the released data. Exact camera-ready figure generation (layout, fonts,
TikZ assembly) is not part of the release contract.

Two paths exist. The record path recomputes the reported quantities from
the released episode records and needs only Python with numpy, scipy,
and matplotlib. The simulation path regenerates the records themselves
and needs Isaac Lab with a GPU.

## From released records, no Isaac

```bash
python scripts/reproduce.py all
```

Every reported quantity maps to a released record and a public command,

| reported quantity | source record | command |
|---|---|---|
| per-rate success fractions and Wilson intervals, all four actors, incl. the upper bound at zero successes | `data/records/eval_summary.jsonl` | `reproduce.py envelope` |
| expert over ACT-A matched gap at r=2 (0.31) and its 20000-resample paired bootstrap 95% interval ([0.18, 0.44]) | `{expert,act}_episodes.jsonl.gz` | `reproduce.py envelope` (or `plot_envelope.py --gap expert act`) |
| stage-completion and terminal-failure counts per actor and rate | `eval_summary.jsonl`, episode records | `reproduce.py stages`, fields per episode |
| ACT-B/C replication cells and degradation r=1 to r=2 | `data/records/replication/summary_seed{1_rerun,2,3}.jsonl` and episode records | direct inspection, `summarize_act_multiseed.py` |
| demonstration-scaling cells (n=50/100/297) | `data/records/replication/summary_n{50,100}.jsonl`, `eval_summary.jsonl` | direct inspection |
| expert calibration counts, eleven rates 0.5 through 6.0 | `experiments/paper/results/expert_sweep_summary.jsonl` | direct inspection |
| expert high-rate ceiling statistics, arm joint-velocity utilization, tracking error | `expert_episodes.jsonl.gz` | `reproduce.py expert-ceiling` |
| hand-object relative-motion summaries (slip per phase) | episode records, `eval_summary.jsonl` | fields per episode, `reproduce.py stages` inputs |
| relative-motion handoff denominators and outcomes | `experiments/paper/results/relative_handoff_summary.jsonl`, `handoff_summary.jsonl` | direct inspection, `summarize_relative_handoff.py` |
| force-closure counts and the one-sided acquisition result | episode records | `reproduce.py certificate` |
| held-out force-closure ranking and calibration statistics | main plus replication episode records | `reproduce.py certificate-oos` |

The learned-policy evaluation grid is the seven rates {0.5, 1, 1.5, 2,
2.25, 2.5, 3}. The expert-only calibration preceding it spans eleven
candidate rates, 0.5 through 6.0, recorded in
`expert_sweep_summary.jsonl` with 64 episodes per rate.

Expected primary numbers, expert and ACT-A both 100/100 at r=1, expert
84/100 and ACT-A 53/100 at r=2, matched gap 0.31 with the
20000-resample paired bootstrap 95% interval [0.18, 0.44].

## Tested environment

The released records and every verification in this repository ran under
the versions below. The simulation stack comes from an Isaac Lab
installation, the analysis tier needs only the three Python packages.

| component | tested version |
|---|---|
| Isaac Sim | 5.1.0 |
| Isaac Lab | 0.54.2 |
| Python (Isaac environment) | 3.11.14 |
| PyTorch | 2.7.0+cu128 |
| SciPy (Isaac environment) | 1.15.3 |
| numpy (Isaac environment) | 1.26.4 |
| diffusers (DP baseline) | 0.30.3 |
| gymnasium | 1.2.1 |

SciPy 1.15.3 is the version behind the frozen Ferrari-Canny
diagnostics. The in-repo scorer reproduces all 6322 recorded margins
exactly under it, while a different SciPy/Qhull build (checked with
system SciPy 1.16.1) flips 15 of 6322 values at the force-closure
boundary, every one with |epsilon| under 1e-12 against the -1 sentinel.
The tie is a numerical-version fact of the boundary, not an
implementation difference, and rescoring for comparison against the
released margins should pin SciPy 1.15.3.

## Full simulation reproduction, Isaac plus GPU

Prerequisites, an Isaac Lab installation matching the tested versions
above, the extension installed into its environment,

```bash
uv pip install -p <isaaclab-venv>/bin/python -e source/parcelstow
# add [diffusion] for the Diffusion Policy baseline, [analysis] for the
# record-reproduction tier, [all] for everything
```

Then, in dependency order,

| step | command | rough cost |
|---|---|---|
| demonstrations | `scripts/manipulation/run_stow_expert.py --record` | under 1 h |
| ACT training | `scripts/manipulation/run_stow_act.py` | about 45 min |
| DP training | `scripts/manipulation/run_stow_diffusion_policy.py` | hours |
| DAgger training | `scripts/manipulation/run_stow_distill.py` | about 1 h |
| main evaluation | `scripts/evaluate.py --actor expert act dp dagger --episodes 100` | about 2 h |
| handoff diagnostics | `scripts/manipulation/stow_relative_handoff.py` | about 1 h |
| videos | `scripts/manipulation/record_stow_rollouts.py` | minutes |

Costs are from a single RTX 5070 Ti with 32 simulator environments.
`scripts/isaac_run.sh <logfile> <command...>` runs any of these with
throttled cores and logging, set `ISAACLAB_VENV` to the Isaac Lab
environment first.

Training seeds, demonstration subsets, and the evaluation seed law are
frozen in the drivers' defaults, a rerun reproduces the released records
up to simulator nondeterminism.

## Anchors to the frozen protocol

- evaluation draws, seed 12345 + 1000 x rate-index, shared across actors
- rate grid, r in {0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0}
- 100 episodes per condition, corruption off, jitter 10 mm
- ACT-A demonstrations, 297 episodes over r in [0.5, 2.0]
