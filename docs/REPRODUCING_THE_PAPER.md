# Reproducing the paper

Two paths exist. The record path reruns every analysis from the released
episode records and needs only Python with numpy and matplotlib. The
simulation path regenerates the records themselves and needs Isaac Lab
with a GPU.

## From released records, no Isaac

```bash
python scripts/reproduce.py all
```

| target | paper output | source records | written to |
|---|---|---|---|
| `envelope` | operating-envelope figure and table, paired bootstrap interval of the expert over ACT-A gap at r=2 | `data/records/eval_summary.jsonl`, `{expert,act}_episodes.jsonl.gz` | `media/operating_envelope.*`, stdout |
| `stages` | stage-completion against rate per actor | `data/records/eval_summary.jsonl` | `media/stages_vs_rate.*` |
| `certificate` | realized-contact force-closure analysis, the one-sided acquisition result | episode records | `outputs/reproduce/certificate_analysis.*` |
| `expert-ceiling` | tracking-accuracy attribution of the expert rate limit | `expert_episodes.jsonl.gz` | `outputs/reproduce/expert_ceiling_analysis.*` |

Further released analyses live precomputed under
`experiments/paper/results/` (ACT seed replication, demonstration
scaling, relative-motion handoff, out-of-sample certificate analysis,
rate-conditioned margin), with their producing scripts under
`scripts/manipulation/analyze_*.py` and `summarize_*.py`, all runnable
from the records in `data/records/` and `experiments/paper/results/`.

Expected primary numbers, expert and ACT-A both 100/100 at r=1, expert
84/100 and ACT-A 53/100 at r=2, paired bootstrap 95% interval of the gap
at r=2 equal to [0.18, 0.44].

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
