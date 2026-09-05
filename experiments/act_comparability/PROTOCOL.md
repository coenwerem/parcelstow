# ACT Training and Evaluation Protocol

## Status and Scope

This protocol governs new upright-placement and keyed-peg-insertion ACT
development. It does not replace current checkpoints or evaluation records,
and it does not modify parcel-insertion v1.0.0 data. The existing upright and
peg records have already been inspected and cannot serve as an untouched final
evaluation for a new model.

The protocol must be committed before an expensive training run begins. Any
change to the factors, selection rule, or reporting rule starts a new protocol
revision and invalidates final evaluation performed under the earlier version.

## Hypotheses

The bounded study tests four ordered hypotheses:

1. Correctly reducing reconstruction loss over valid action elements improves
   late-episode supervision.
2. Episode-level validation and periodic checkpoint retention prevent an
   arbitrary final epoch from defining the policy.
3. Peg acquisition outside its demonstrated speed range is sensitive to the
   raw rate input during fixed-duration acquisition phases.
4. If the preceding corrections do not close the nominal gap, the remaining
   difference reflects demonstration coverage, ACT chunk execution, or task
   difficulty rather than a verified interface defect.

The rate-input intervention is diagnostic only. A policy reported in the
benchmark continues to receive the documented raw speedup factor.

## Demonstration Partition

For the first bounded pilot, use the current task-specific demonstration file
and record its SHA-256 checksum. Split complete episodes with NumPy's
`default_rng(20260904).permutation`:

- 80% of admitted episodes form the training partition;
- the remaining 20% form the action-validation partition;
- normalization statistics use training episodes only;
- no timestep from a validation episode enters optimization.

The partition indices, source path, source checksum, admitted count, and
rejected demonstration summary must be stored with each run. The pilot varies
only the corrected loss and checkpoint-selection procedure. It does not
collect more demonstrations.

If the pilot shows unresolved coverage sensitivity, a later protocol revision
may collect a new indexed demonstration set. That revision must predeclare the
rate distribution, object and robot initial-state bank, attempted episode
count, success-filtering rule, and treatment of rejected expert episodes.

## Optimization and Checkpoint Candidates

Retain the existing ACT architecture and optimizer settings. Train for at most
2,000 epochs with policy seeds 0, 1, and 2. The split seed remains fixed across
policy seeds. Reduce L1 over valid action elements only, so padding does not
change the reconstruction scale.

Every 100 epochs, evaluate deterministic zero-latent action chunks on fixed
validation start indices. Record valid-element L1, first-action L1, KL loss
under teacher forcing, epoch, elapsed time, and the training loss components.
Retain the five candidates with the lowest deterministic validation
valid-element L1 for each policy seed, plus the last checkpoint. Validation
start indices are generated once from seed 20260905 and stored in the run
configuration.

This retention rule limits each policy seed to six approximately 291 MB model
files. Expected storage is at most 1.75 GB per seed, excluding logs, and the
observed training runtime is approximately 45 to 55 minutes per 2,000-epoch
run on the recorded RTX 5070 Ti system.

## Model-Development Evaluation

Evaluate each retained candidate at `r=1` on 100 indexed episodes with 10 mm
planar jitter, 32 environments, no observation corruption, and base evaluation
seed 42001. This bank is used for model development and must never be reported
as final evidence.

For each policy seed, select the candidate by:

1. highest task-success count;
2. highest lexicographic task-specific stage-count vector in the documented
   stage order;
3. lowest deterministic validation valid-element L1;
4. earliest epoch.

Policy seed 0 is the predeclared primary ACT instance. Seeds 1 and 2 are
replication instances and must be reported separately. A seed cannot replace
seed 0 because its development score is higher.

The nominal development command for a selected checkpoint is:

```bash
python scripts/evaluate.py --task TASK --actor act --rates 1.0 \
  --episodes 100 --num_envs 32 --eval_seed 42001 \
  --out_dir outputs/act_comparability/TASK/seed-SEED/development \
  --act_ckpt CHECKPOINT --tag _epoch-EPOCH
```

Replace `TASK`, `SEED`, `CHECKPOINT`, and `EPOCH` with recorded values. The
task-specific driver accepts `--act_ckpt` through the public wrapper's
passthrough arguments.

## Final Evaluation Bank

Do not generate or inspect the final bank until the training settings,
candidate-retention rule, development-selection rule, nominal comparison
criterion, and selected checkpoint hashes are frozen. The final base seed is
73001. Evaluate 200 indexed episodes per actor and speed with 10 mm planar
jitter, 32 environments, and no observation corruption.

Use the current task grids:

- upright placement: `0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.5`;
- keyed peg insertion: `0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.5`.

The final command for each task and selected policy seed is:

```bash
python scripts/evaluate.py --task TASK --actor expert act \
  --rates 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.5 \
  --episodes 200 --num_envs 32 --eval_seed 73001 \
  --out_dir outputs/act_comparability/TASK/seed-SEED/final \
  --act_ckpt CHECKPOINT --tag _protocol-1
```

Run final evaluation once per selected checkpoint. Do not use a final result to
change training, select a seed, select a checkpoint, alter the task grid, or
revise a success predicate.

## Nominal Comparison Criterion

The recommended criterion is paired non-inferiority at `r=1`: the lower
one-sided 95% confidence bound for `p_ACT - p_expert` must exceed -0.10 on the
200 matched final episodes. Report both success proportions, the paired
difference, its confidence interval, and the discordant-pair counts.

This margin permits ACT to succeed on as many as ten percentage points fewer
episodes than the expert, subject to sampling uncertainty. It therefore means
nominal comparability, not equality. The author must approve or replace the
margin and confidence procedure before this criterion governs the consolidated
study. Until that decision is recorded, use `nominal comparison pending`
rather than `parity` in retained results.

## Fixed-Phase Rate Diagnostic

For keyed peg insertion, first evaluate the current checkpoint on the same 40
indexed episodes at actual `r=1.5` under two policy inputs:

- documented observation, including raw `r=1.5`;
- rate feature replaced by 1.0 only while the task-specific phase index is 0
  through 4.

All simulator timing, physics, task definitions, initial conditions, and other
observation values remain unchanged. Recovery of acquisition under the second
condition would isolate policy sensitivity to an out-of-range rate feature.
It would not justify changing the benchmark observation or reporting the
intervened policy as a baseline.

Write the two record sets beneath
`outputs/act_comparability/peg/rate-input-diagnostic/` with distinct filenames
and record the rate replacement in every episode and summary row.

## Required Record Fields

Each retained run must identify:

- task and Gym identifier;
- policy name and actor specification;
- policy seed and checkpoint epoch;
- checkpoint path and SHA-256 checksum;
- source demonstration path and SHA-256 checksum;
- training and validation episode indices;
- split seed and validation-start seed;
- architecture and optimizer configuration;
- Git commit and environment versions;
- task rate, evaluation base seed, per-rate seed, and bank role;
- initial-condition identifier and initial-condition-bank checksum;
- number of environments, requested episodes, jitter, and corruption state;
- task-specific stage outcomes and terminal failure reason;
- physical task-success result.

Training logs must also include reconstruction, KL, total training, and
validation losses. Simulator summaries must retain the paired discordant
counts when expert and ACT share a bank.

## Stopping Rules

Stop a pilot when any of the following occurs:

- the 2,000-epoch budget is reached;
- loss becomes nonfinite;
- a checkpoint or data checksum changes unexpectedly;
- a task, observation, action, or initial-condition definition differs from
  the committed protocol;
- the development bank has been used to choose an unlisted hyperparameter;
- the final bank is inspected before policy selection is frozen.

The first pilot runs only seed 0 for each task. It tests execution and effect
size, not replication or final comparability. Proceed to seeds 1 and 2 only if
the pilot is reproducible and its development evidence supports the corrected
procedure.

## Deferred Bimanual Asset Review

After both upright placement and keyed peg insertion satisfy the author-approved
nominal ACT comparison criterion, review assets for a future bimanual
manipulation system based on a RealHand A7 or P7 arm configuration. The source
organization is [RealHand-Robotics](https://github.com/RealHand-Robotics).
As checked on 2026-09-04, its
[`XRoboToolkit_Realhand_Dexterous_Hand_Example`](https://github.com/RealHand-Robotics/XRoboToolkit_Realhand_Dexterous_Hand_Example)
repository contains dual-A7 URDF and mesh assets. No public P7 asset was found
by name in the organization at that time. Before importing anything, verify
the asset license, kinematic and inertial data, collision geometry, mesh units,
joint limits, actuator semantics, provenance, and Isaac Lab conversion path.

This review is deferred work. It does not establish hardware compatibility,
simulation validity, or a new ParcelStow task.
