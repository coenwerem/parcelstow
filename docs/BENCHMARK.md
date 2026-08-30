# Benchmark specification

ParcelStow instantiates a matched expert-learner evaluation of temporal
robustness in dexterous manipulation. It measures task-success probability
as a function of task execution speed while everything else about the task
stays fixed, and compares the expert to each learner at the same speeds.

## The controlled variable

The speedup factor r scales the duration of the manipulation phase
schedule that runs after the parcel has been acquired. r = 1 reproduces
the demonstrated cycle (14.1 s), r = 2 halves the manipulation-phase
durations (10.2 s including the rate-fixed acquisition segment) and is
the maximum demonstrated speed, the upper boundary of the demonstrated
speed range. The acquisition timing, the grasp, the object, the geometry,
the success predicates, and the observation and action interfaces do not
change with r. Every policy observes r, so it always knows the speed
demanded of it. [TASK_SPEC.md](TASK_SPEC.md) freezes every constant.

## What stays fixed as r changes

- object, an 80 x 55 x 40 mm, 0.120 kg cuboid, friction 0.5
- receptacle, an open-front receptacle with 10 mm entrance clearance per side
- grasp, one five-contact grasp from the frozen bank
- expert path, lift 80 mm, reorient 90 deg, translate, insert, release
- start-pose law, x-y jitter of 10 mm about the frozen start, yaw fixed
- evaluation draws, the same seeds and start poses for every policy
- success, the physical predicate chain of TASK_SPEC section 8

## Success

An episode succeeds only if every stage predicate holds through settling,
acquisition with force-bearing contacts, lift clear of the table,
reorientation past the tolerance, insertion to depth inside the
receptacle, release, and a settled final pose inside position and
orientation tolerances (50 mm, 10 deg). The predicates read physical
state only, no analytic grasp metric enters any predicate.

## Evaluation protocol

- 100 episodes per policy per speed, batched over 32 environments
- shared draws, seed 12345 + 1000 x speed-index, identical across policies
- corruption off, jitter 10 mm
- speedup grid of the paper, r in {0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0}
- ACT training range, demonstrations span r in [0.5, 2.0], so r = 2.0
  sits at the boundary of the demonstrated range and r > 2 extrapolates
- reported intervals, Wilson 95% per condition, paired bootstrap for
  matched expert-learner differences on the shared draws

## Expert and learners

| policy | construction | role in the measurement |
|---|---|---|
| expert | model-derived scripted policy, IK-tracked object path with grasp-bank acquisition | supplies the demonstrations and the task-success curve every learner is compared against at matched speeds |
| act | state-only ACT (Zhao et al., RSS 2023) trained on 297 expert demonstrations, seeds A/B/C | reaches nominal parity, 100/100 at r=1 for seed A, so its task-success curve is directly comparable to the expert's across the grid |
| dp | state-based Diffusion Policy (ConditionalUnet1D) on the same demonstrations | a second architecture on the same demonstrations, 68/100 at r=1, below nominal parity |
| dagger | MLP student distilled with DAgger | 3/100 at r=1 and below parity at every speed, the control for what the measurement cannot separate |

## Diagnostics beside the task-success curves

The episode records carry stage outcomes, in-hand slip, hand-object
relative motion, actuator utilization, realized contact sets with
force-closure margins, and the tracking error against the commanded
targets. [DIAGNOSTICS.md](DIAGNOSTICS.md) documents each field and the
one-sided role of acquisition-time force closure.

## Using the benchmark on a new policy

[POLICY_INTERFACE.md](POLICY_INTERFACE.md) defines the actor interface.
The one command

```bash
python scripts/evaluate.py --actor your.module:YourPolicy \
    --rates 0.5 1.0 1.5 2.0 2.25 2.5 3.0 --episodes 100
```

produces records in the released schema, and

```bash
python scripts/plot_envelope.py --summary outputs/eval/summary.jsonl
```

draws the task-success curve next to the released expert and learners.
