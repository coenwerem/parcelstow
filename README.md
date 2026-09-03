# ParcelStow

Isaac Lab robot learning benchmark for expert–learner evaluation across task execution speeds.

[![CI](https://github.com/coenwerem/parcelstow/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/coenwerem/parcelstow/actions/workflows/ci.yml)
[![Apache-2.0 License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-0.54.2-76B900?logo=nvidia&logoColor=white)](https://isaac-sim.github.io/IsaacLab/)
[![arXiv:2609.01453](https://img.shields.io/badge/arXiv-2609.01453-b31b1b.svg)](https://arxiv.org/abs/2609.01453)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97-Dataset-yellow.svg)](https://huggingface.co/datasets/cenwerem/parcelstow)

ParcelStow evaluates a scripted expert and learned imitation policies (hereafter, *learner*) under matched initial conditions and at varying task execution speeds, with the execution speed characterized by a positive scalar speedup factor `r`. The expert and learner use the same task geometry, physical success predicates, state observation, and joint-position action interface at each evaluated speed. ParcelStow contains three tasks: parcel insertion, upright placement, and keyed peg insertion (see the gallery below and [Task Results](#current-three-task-results)).

The stable [`v1.0.0`](https://github.com/coenwerem/parcelstow/releases/tag/v1.0.0) release corresponds to the parcel-insertion study in [arXiv:2609.01453](https://arxiv.org/abs/2609.01453). The software on `main` is in active development and contains all three tasks. A later v2 release is underway.

## Task Gallery
<table align="center">
  <tr>
    <td align="center" valign="top" width="33%">
      <a href="https://huggingface.co/datasets/cenwerem/parcelstow/resolve/main/videos/parcel_expert_r2_2x_faster.mp4">
        <img src="media/parcel_expert_r2_2x_faster.gif" alt="Expert parcel insertion at r=2" width="100%">
      </a>
      <br>
      <sub><b>Parcel Insertion.</b> Acquire and reorient a parcel, then insert it into an open-front receptacle. The full-resolution video is hosted on Hugging Face.</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <a href="https://huggingface.co/datasets/cenwerem/parcelstow/resolve/main/videos/upright_expert_r1_2x_faster.mp4">
        <img src="media/upright_expert_r1_2x_faster.gif" alt="Expert upright placement at r=1" width="100%">
      </a>
      <br>
      <sub><b>Upright Placement.</b> Reorient a cuboid to an upright pose and release it in a target region. The full-resolution video is hosted on Hugging Face.</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <a href="https://huggingface.co/datasets/cenwerem/parcelstow/resolve/main/videos/peg_expert_r1_2x_faster.mp4">
        <img src="media/peg_expert_r1_2x_faster.gif" alt="Expert keyed peg insertion at r=1" width="100%">
      </a>
      <br>
      <sub><b>Keyed Peg Insertion.</b> Reorient a cuboid and insert it into a square pocket with 3 mm of clearance per side. The full-resolution video is hosted on Hugging Face.</sub>
    </td>
  </tr>
</table>

## Reproduce Results from Evaluation Records
The compressed evaluation records in `data/records/` can be analyzed without
Isaac Lab. Install NumPy and Matplotlib in a Python 3 environment, then
recompute the success counts, tables, and figures for all three tasks:

```bash
python3 -m pip install numpy matplotlib
python3 scripts/reproduce.py all-tasks
```

The parcel records belong to v1.0.0; the upright and peg records describe the
active development branch. This analysis does not require a GPU, checkpoint,
or demonstration file. The existing parcel-paper targets remain available
through `python3 scripts/reproduce.py all`. See [Reproducing the
Results](docs/REPRODUCING_THE_PAPER.md) for the source record behind each
result.

## Get the Code

Clone `main` to use the current three-task benchmark:

```bash
git clone --branch main https://github.com/coenwerem/parcelstow.git
cd parcelstow
```

To reproduce the stable parcel-insertion software associated with arXiv v1,
check out the `v1.0.0` release instead:

```bash
git clone --branch v1.0.0 --depth 1 https://github.com/coenwerem/parcelstow.git parcelstow-v1.0.0
cd parcelstow-v1.0.0
```

The `v1.0.0` release does not contain the upright-placement or keyed-peg tasks.

## Installation
Simulator execution requires Isaac Lab and a supported NVIDIA GPU. From the
repository root, install the extension into the Isaac Lab Python environment:

```bash
uv pip install -p <isaaclab-venv>/bin/python -e source/parcelstow
```

The provided records were produced with Python 3.11.14, Isaac Sim 5.1.0, Isaac Lab 0.54.2, PyTorch 2.7.0+cu128, SciPy 1.15.3, NumPy 1.26.4, and one NVIDIA RTX 5070 Ti. CPU-only record reproduction supports Python 3.10 or later.

Run direct module tests without Isaac Lab:

```bash
python -m pytest tests/ -q
```

Run the simulator groups in separate processes:

```bash
python -m pytest tests/test_parcel_physics.py tests/test_relative_handoff.py --isaac -q
python -m pytest tests/test_upright_physics.py --isaac-upright -q
python -m pytest tests/test_peg_physics.py --isaac-peg -q
```

## Run a Task
Change only `--task` to run another scripted expert:

```bash
python scripts/run_task.py --task parcel
python scripts/run_task.py --task upright
python scripts/run_task.py --task peg
```

Omitting `--task` preserves the `v1.0.0` parcel-insertion behavior: `python scripts/run_task.py`.

Evaluate the released experts through the same public interface:

```bash
python scripts/evaluate.py --task parcel --actor expert
python scripts/evaluate.py --task upright --actor expert
python scripts/evaluate.py --task peg --actor expert
```

The `--task` value selects one task: `parcel` selects parcel insertion,
`upright` selects upright placement, and `peg` selects keyed peg insertion.
Download the selected task's ACT checkpoint and demonstrations with:

```bash
python scripts/download_artifacts.py --task parcel
python scripts/download_artifacts.py --task upright
python scripts/download_artifacts.py --task peg
```

Evaluate each downloaded ACT checkpoint with:

```bash
python scripts/evaluate.py --task parcel --actor act
python scripts/evaluate.py --task upright --actor act
python scripts/evaluate.py --task peg --actor act
```

Parcel insertion also provides Diffusion Policy and DAgger checkpoints through
`python scripts/download_artifacts.py --paper`. Upright placement and keyed peg
insertion do not provide those checkpoints.

## Current Three-Task Results
Each policy-speed evaluation condition contains 100 episodes. The expert and learner use initial conditions indexed by speed and episode. Parcel ACT-A is the primary comparison because ACT-A and the expert both succeed in 100/100 episodes at `r=1`. The upright and peg ACT checkpoints do not meet that nominal-matching condition and are secondary task-specific results on the active development branch.

| Task | Gym Identifier | Demonstrated `r` | Expert at `r=1` | ACT at `r=1` | Additional Current Result |
|---|---|---:|---:|---:|---|
| Parcel insertion | `ParcelStow-L6-Distill-Play-v0` | `[0.5, 2.0]` | 100/100 | ACT-A 100/100 | at `r=2`: expert 84/100, ACT-A 53/100 |
| Upright placement | `UprightPlace-L6-Play-v0` | `[0.75, 1.75]` | 92/100 | ACT 39/100 | at `r=1.75`: expert 90/100, ACT 74/100 |
| Keyed peg insertion | `PegInsert-L6-Play-v0` | `[0.5, 1.0]` | 93/100 | ACT 75/100 | ACT acquisition is 0/100 at each evaluated `r >= 1.5` |

### Expert–ACT Rollouts at `r=2`
Each video compares the expert and ACT on one matched initial condition. These
episodes illustrate task outcomes; the table above reports the 100-episode
evaluation results.

<table align="center">
  <tr>
    <td align="center" valign="top" width="33%">
      <a href="https://huggingface.co/datasets/cenwerem/parcelstow/resolve/main/videos/parcel_expert_vs_act_r2_2x_faster.mp4">
        <img src="media/parcel_expert_vs_act_r2_2x_faster.gif" alt="Expert and ACT parcel insertion rollouts at r=2" width="100%">
      </a>
      <br>
      <sub><b>Parcel Insertion.</b> The expert succeeds; ACT finishes with an orientation error above the success threshold.</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <a href="https://huggingface.co/datasets/cenwerem/parcelstow/resolve/main/videos/upright_expert_vs_act_r2_2x_faster.mp4">
        <img src="media/upright_expert_vs_act_r2_2x_faster.gif" alt="Expert and ACT upright placement rollouts at r=2" width="100%">
      </a>
      <br>
      <sub><b>Upright Placement.</b> The expert succeeds; the cuboid placed by ACT tips after release.</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <a href="https://huggingface.co/datasets/cenwerem/parcelstow/resolve/main/videos/peg_expert_vs_act_r2_2x_faster.mp4">
        <img src="media/peg_expert_vs_act_r2_2x_faster.gif" alt="Expert and ACT keyed peg insertion rollouts at r=2" width="100%">
      </a>
      <br>
      <sub><b>Keyed Peg Insertion.</b> The expert succeeds; ACT does not acquire the peg.</sub>
    </td>
  </tr>
</table>

The [Benchmark Specification](docs/BENCHMARK.md) defines matched evaluation. The task specifications freeze each task's geometry, phase schedule, initial-condition distribution, stage outcomes, failure reasons, and physical success predicates:

- [Parcel Insertion Task Specification](docs/TASK_SPEC.md)
- [Upright Placement Task Specification](docs/TASK_SPEC_UPRIGHT.md)
- [Keyed Peg Insertion Task Specification](docs/TASK_SPEC_PEG.md)

## Evaluate A Custom Policy on All Tasks
The same Python class can be loaded for every task:

```bash
python scripts/evaluate.py --task parcel --actor examples.custom_policy:HoldPosturePolicy --rates 1 --episodes 5
python scripts/evaluate.py --task upright --actor examples.custom_policy:HoldPosturePolicy --rates 1 --episodes 5
python scripts/evaluate.py --task peg --actor examples.custom_policy:HoldPosturePolicy --rates 1 --episodes 5
```

All tasks produce a 147-dimensional state observation and accept a 16-dimensional normalized joint-position action at 50 Hz. Task identity is selected by `--task`; it is not appended to the observation. Observation index 146 contains `r`. The pose slice at indices 118:125 represents `parcel_pose` for parcel insertion and `object_pose` for upright placement and keyed peg insertion. Phase values retain task-specific schedules. [Policy Interface](docs/POLICY_INTERFACE.md) documents every slice and the adapter boundary.

`HoldPosturePolicy` commands the default posture and normally fails. It demonstrates loading and record generation, not task performance.

## Data, Checkpoints, and Videos
The [Hugging Face dataset](https://huggingface.co/datasets/cenwerem/parcelstow) hosts Parquet demonstrations for interactive loading, `.pt` demonstrations used by training scripts, task-specific checkpoints, and videos. GitHub stores the frozen v1 parcel records, current development records for upright and peg, and CPU-only reproduction code. [`artifacts/manifest.json`](artifacts/manifest.json) preserves every hosted path, byte count, and SHA-256 checksum.

See [Data and Checkpoints](docs/DATA_AND_CHECKPOINTS.md) for the file map and checkpoint limitations.

## Contributing
[Contributing](CONTRIBUTING.md) distinguishes bug reports, policy results, policy integrations, candidate tasks, and changes to fixed definitions. [Candidate Task Authoring Protocol](docs/TASK_AUTHORING.md) defines the scientific and software evidence required before a task can be listed as part of ParcelStow.

## Repository Map
| Path | Content |
|---|---|
| `scripts/run_task.py`, `scripts/evaluate.py` | public simulator commands for all three tasks |
| `scripts/reproduce.py` | CPU-only numerical reproduction |
| `scripts/task_registry.py` | task aliases, gym IDs, defaults, stage keys, experts, monitors, and schedules |
| `source/parcelstow/` | Isaac Lab extension and task definitions |
| `data/records/` | frozen v1 parcel records and current development records for upright and peg |
| `examples/custom_policy.py` | one policy class loadable on all tasks |
| `docs/` | current benchmark, policy, reproduction, contribution, and task specifications |

## Citation
The current arXiv v1 reports the parcel-insertion study:

```bibtex
@misc{enwerem2026parcelstow,
  title         = {Does Imitation Learning Preserve Temporal Robustness in Dexterous Manipulation? An Expert-Learner Comparison Across Task Execution Speeds},
  author        = {Enwerem, Clinton and Baras, John S. and Belta, Calin},
  year          = {2026},
  eprint        = {2609.01453},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2609.01453}
}
```

Use `CITATION.cff` for the stable `v1.0.0` software citation. The file records the latest software release; changes on `main` have not been released.
