# Reproducing the Results

The stable v1.0.0 release and arXiv v1 cover parcel insertion. Its frozen
episode records and analysis code reproduce the parcel results reported in
the paper. The active development branch also contains evaluation records
for upright placement and keyed peg insertion. Those two tasks are not part
of v1.0.0 or arXiv v1.

Two paths exist. The record path recomputes the v1 parcel quantities and the
current three-task development results from the provided episode records; it
needs only Python with NumPy, SciPy, and Matplotlib. The simulation path
regenerates the records themselves and needs Isaac Lab with a GPU.

## CPU-Only Reproduction from Records

```bash
python scripts/reproduce.py all-tasks
python scripts/reproduce.py all
```

`all-tasks` prints the source path and demonstrated-range boundaries for the
three current tasks. It writes one task-specific success table and figure per
task. `all` reproduces the parcel analyses associated with arXiv v1.

Every quantity below maps to a record and a public command:

| reported quantity | source record | command |
|---|---|---|
| current three-task success tables and curves | `data/records/{,upright/,peg/}{expert,act}_episodes.jsonl.gz` | `reproduce.py all-tasks` |
| task-success fractions and Wilson intervals at each speed, including the upper bound at zero successes | `data/records/eval_summary.jsonl` | `reproduce.py envelope` |
| expert over ACT-A matched gap at r=2 (0.31) and its 20000-resample paired bootstrap 95% interval ([0.18, 0.44]) | `{expert,act}_episodes.jsonl.gz` | `reproduce.py envelope` (or `plot_envelope.py --gap expert act`) |
| stage-completion and terminal-failure counts per policy and speed | `eval_summary.jsonl`, episode records | `reproduce.py stages`, fields per episode |
| ACT-B/C replication cells and degradation r=1 to r=2 | `data/records/replication/summary_seed{1_rerun,2,3}.jsonl` and episode records | direct inspection, `summarize_act_multiseed.py` |
| demonstration-scaling cells (n=50/100/297) | `data/records/replication/summary_n{50,100}.jsonl`, `eval_summary.jsonl` | direct inspection |
| expert calibration counts, eleven candidate speeds 0.5 through 6.0 | `experiments/paper/results/expert_sweep_summary.jsonl` | direct inspection |
| expert arm joint-velocity utilization and target-tracking error at each speed | `expert_episodes.jsonl.gz` | `reproduce.py expert-ceiling` |
| hand-object relative-motion summaries (slip per phase) | episode records, `eval_summary.jsonl` | fields per episode, `reproduce.py stages` inputs |
| relative-motion handoff denominators and outcomes | `experiments/paper/results/relative_handoff_summary.jsonl` | direct inspection, `summarize_relative_handoff.py` |
| force-closure counts and the one-sided acquisition result | episode records | `reproduce.py certificate` |
| held-out force-closure ranking and calibration statistics | main plus replication episode records | `reproduce.py certificate-oos` |

The learned-policy evaluation grid contains the seven speedup factors
`{0.5, 1, 1.5, 2, 2.25, 2.5, 3}`. The expert-only calibration preceding it spans eleven
candidate speeds, 0.5 through 6.0, recorded in
`expert_sweep_summary.jsonl` with 64 episodes per speed.

The expected primary results are 100/100 successes for both the expert and
ACT-A at `r=1`, followed by 84/100 for the expert and 53/100 for ACT-A at
`r=2`. The matched difference is 0.31, with a 20,000-resample paired-bootstrap
95% interval of `[0.18, 0.44]`.

## Tested Environment

The provided records and every verification in this repository ran under
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
| GPU | NVIDIA RTX 5070 Ti |
| diffusers (Diffusion Policy) | 0.30.3 |
| gymnasium | 1.2.1 |

SciPy 1.15.3 is the version behind the frozen Ferrari-Canny
diagnostics. The in-repo scorer reproduces all 6322 recorded margins
exactly under it, while a different SciPy/Qhull build (checked with
system SciPy 1.16.1) flips 15 of 6322 values at the force-closure
boundary, every one with |epsilon| under 1e-12 against the -1 sentinel.
The tie is a numerical-version fact of the boundary, not an
implementation difference, and rescoring for comparison against the
released margins should pin SciPy 1.15.3.

## Full Simulation Reproduction, Isaac Plus GPU

Full simulation reproduction requires an Isaac Lab installation matching the
tested versions above. Install the extension into that environment with

```bash
uv pip install -p <isaaclab-venv>/bin/python -e source/parcelstow
# add [diffusion] for Diffusion Policy, [analysis] for the
# record-reproduction tier, [all] for everything
```

Run simulator test groups in separate processes before regenerating records:

```bash
python -m pytest tests/test_parcel_physics.py tests/test_relative_handoff.py --isaac -q
python -m pytest tests/test_upright_physics.py --isaac-upright -q
python -m pytest tests/test_peg_physics.py --isaac-peg -q
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

These costs were measured on a single RTX 5070 Ti with 32 simulator
environments. `scripts/isaac_run.sh <logfile> <command...>` runs any command
with CPU throttling and logging. Set `ISAACLAB_VENV` to the Isaac Lab
environment before using it.

The driver defaults fix the training seeds, demonstration subsets, and
evaluation seed law. A rerun targets the provided records up to simulator
nondeterminism.

## Anchors to the Frozen Protocol

- evaluation draws, seed 12345 + 1000 x speed-index, shared across policies
- speedup grid, r in {0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0}
- 100 episodes per condition, corruption off, jitter 10 mm
- ACT-A demonstrations, 297 episodes over r in [0.5, 2.0]
