# Benchmark specification

ParcelStow measures one quantity, the task-rate operating envelope of a
manipulation policy, success fraction as a function of the requested task
rate r while everything else about the task stays fixed.

## The controlled variable

The task rate r scales the duration of the manipulation phase schedule.
r = 1 reproduces the demonstrated cycle (14.1 s), r = 2 halves it
(10.2 s including the rate-fixed segments). The acquisition timing, the
grasp, the object, the geometry, the success predicates, and the
observation and action interfaces do not change with r. The requested
rate enters the policy observation, so a policy always knows the demand
placed on it. [TASK_SPEC.md](TASK_SPEC.md) freezes every constant.

## What stays fixed as r changes

- object, an 80 x 55 x 40 mm, 0.120 kg cuboid, friction 0.5
- receptacle, an open-top cubby with 10 mm entrance clearance per side
- grasp, one five-contact grasp from the frozen bank
- expert path, lift 80 mm, reorient 90 deg, translate, insert, release
- start-pose law, x-y jitter of 10 mm about the frozen start, yaw fixed
- evaluation draws, the same seeds and start poses for every actor
- success, the physical predicate chain of TASK_SPEC section 8

## Success

An episode succeeds only if every stage predicate holds through settling,
acquisition with force-bearing contacts, lift clear of the table,
reorientation past the tolerance, insertion to depth inside the cubby,
release, and a settled final pose inside position and orientation
tolerances (50 mm, 10 deg). The predicates read physical state only, no
analytic grasp metric enters any predicate.

## Evaluation protocol

- 100 episodes per actor per rate, batched over 32 environments
- shared draws, seed 12345 + 1000 x rate-index, identical across actors
- corruption off, jitter 10 mm
- rate grid of the paper, r in {0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0}
- ACT training range, demonstrations span r in [0.5, 2.0], so r = 2.0
  sits inside the demonstrated range and r > 2 extrapolates
- reported intervals, Wilson 95% per condition, paired bootstrap for
  cross-actor gaps on the shared draws

## Reference actors

| actor | description |
|---|---|
| expert | model-derived scripted policy, IK-tracked object path with grasp-bank acquisition |
| act | state-only ACT (Zhao et al., RSS 2023) trained on 297 expert demonstrations, seeds A/B/C |
| dp | state-based Diffusion Policy (ConditionalUnet1D) on the same demonstrations |
| dagger | MLP student distilled with DAgger, a deliberately weak reference |

## Diagnostics beside the envelope

The episode records carry stage outcomes, in-hand slip, hand-object
relative motion, actuator utilization, realized contact sets with
force-closure scores, and the tracking error against the commanded
targets. [DIAGNOSTICS.md](DIAGNOSTICS.md) documents each field and the
one-sided role of the force-closure certificate.

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

draws the operating envelope next to the released baselines.
