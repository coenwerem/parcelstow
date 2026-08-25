# Policy interface

ParcelStow evaluates any policy exposing a three-member actor interface.
The reference actors (scripted expert, ACT, Diffusion Policy, DAgger) use
the same interface, defined in `scripts/manipulation/stow_runtime.py` and
frozen by section 10 of [TASK_SPEC.md](TASK_SPEC.md).

## The actor interface

```python
class MyPolicy:
    name = "my_policy"                # stored in every episode record

    def __init__(self, base, checkpoint=None, num_envs=None):
        # base is the Isaac Lab ManagerBasedRLEnv, checkpoint the
        # --custom_ckpt path or None, num_envs the batch width
        ...

    def reset(self, ids, obs=None):
        # ids lists the environment indices being reset, clear any
        # per-env recurrent state (history buffers, action queues)
        ...

    def act(self, obs):
        # obs is a (n, 147) torch tensor on the environment device
        # returns (action, q_target)
        #   action    (n, 16) normalized joint-position command
        #   q_target  (n, 16) absolute joint targets, or None, used only
        #             by the tracking-error diagnostic
        ...
```

A working example lives at `examples/custom_policy.py`. Run it with

```bash
python scripts/evaluate.py --actor examples.custom_policy:HoldPosturePolicy \
    --rates 1.0 --episodes 5 --num_envs 8
```

Any `module.path:ClassName` reachable on the Python path works the same
way, `--custom_ckpt` passes a checkpoint path through to the constructor.

## Observation, 147-D state vector

Slices in order, all quantities from the simulator state after the noise
model of the evaluation protocol (corruption off during evaluation),

| slice | width | content |
|---|---|---|
| 0:51 | 51 | joint positions of all 51 robot joints, relative to defaults |
| 51:102 | 51 | joint velocities, relative |
| 102:118 | 16 | previous action |
| 118:125 | 7 | parcel pose in the pelvis frame, position xyz then quaternion wxyz |
| 125:140 | 15 | distal phalanx positions in the pelvis frame, five fingers times xyz |
| 140:145 | 5 | distal contact force magnitudes, clipped at 50 N, scaled by 1/10 |
| 145 | 1 | task phase, (k + f) / N_PHASES in [0, 1] |
| 146 | 1 | requested task rate r |

The requested rate reaches the policy only through `obs[:, 146]`, a
rate-aware policy reads it there.

## Action, 16-D joint-position target

The action commands the 16 actuated joints of the control chain (waist
yaw, roll, pitch, right shoulder pitch, roll, yaw, elbow, wrist roll,
pitch, yaw, thumb cmc roll, thumb cmc pitch, and the index, middle, ring,
pinky mcp pitch), in `CHAIN_ACTUATED` order. The environment applies

```
joint_target = 0.5 * action + q_default
```

through implicit PD actuators at 50 Hz control (physics at 200 Hz,
decimation 4). An action of zero holds the default posture.

## Episode protocol

- `reset(ids, obs)` runs for every environment at episode start, the
  evaluator batches 32 environments by default.
- One episode spans one full manipulation cycle at the drawn rate plus
  the settling window, `geometry.cycle_time(r)` maps rate to cycle
  seconds per the frozen phase schedule of the task specification.
- The episode record stores the eight stage predicates (acquired,
  lifted_clear, reoriented, preinsert_reached, inserted, released,
  settled, task_success), the failure reason, slip and contact
  diagnostics, and the configuration stamp,
  [DIAGNOSTICS.md](DIAGNOSTICS.md) documents them.
- Identical evaluation seeds produce identical start draws across actors,
  so per-episode comparisons pair by the `episode` field.

## Scientific constraints

The interface is part of the frozen benchmark definition. Comparisons
against the released numbers require the 147-D observation, the 16-D
action, the requested rate inside the observation, and 50 Hz control,
unchanged. A policy built on a different interface can still run in the
environment, but its numbers no longer compare against the released
envelope.
