"""Per-environment task phase and task rate of the ParcelStow task.

The phase clock derives from the per-environment step counter and the
per-environment task rate. Time t = steps * dt runs through the phase
schedule of geometry.PHASES with the manipulation phases divided by the
rate r, so two environments at different rates sit at different phases at
the same wall time. The rate buffer lives on the environment and the reset
event samples it from RATE_SPEC, which the drivers set before every reset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .. import geometry as G

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Driver-set specification of the task-rate law applied at reset.
#   {"mode": "fixed", "value": r}
#   {"mode": "uniform", "lo": a, "hi": b}
#   {"mode": "per_env", "values": tensor (E,)}
RATE_SPEC = {"mode": "fixed", "value": 1.0}


def rate_buf(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "_stow_rate"):
        env._stow_rate = torch.ones(env.num_envs, device=env.device)
    return env._stow_rate


def _durations(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(E, N_PHASES) durations at each environment's rate."""
    if not hasattr(env, "_stow_nominal"):
        env._stow_nominal = torch.tensor(G.NOMINAL_DURATIONS, dtype=torch.float32, device=env.device)
        env._stow_scaled = torch.tensor(G.RATE_SCALED, device=env.device)
    r = rate_buf(env).unsqueeze(1)
    nom = env._stow_nominal.unsqueeze(0).expand(env.num_envs, -1)
    return torch.where(env._stow_scaled.unsqueeze(0), nom / r, nom)


def elapsed_time(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.episode_length_buf.float() * float(env.step_dt)


def phase_state(env: ManagerBasedRLEnv):
    """Returns (k, f, t, cycle) as tensors (E,), phase index (long),
    in-phase fraction, elapsed time, and the cycle time at the env's rate."""
    d = _durations(env)
    cum = torch.cumsum(d, dim=1)
    t = elapsed_time(env)
    k = (t.unsqueeze(1) >= cum).sum(dim=1).clamp(max=G.N_PHASES - 1)
    start = torch.where(k > 0, torch.gather(cum, 1, (k - 1).clamp(min=0).unsqueeze(1)).squeeze(1),
                        torch.zeros_like(t))
    dur = torch.gather(d, 1, k.unsqueeze(1)).squeeze(1)
    f = ((t - start) / dur).clamp(0.0, 1.0)
    return k, f, t, cum[:, -1]


def task_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Observation, (k + f) / N_PHASES in [0, 1], shape (E, 1)."""
    k, f, _, _ = phase_state(env)
    return ((k.float() + f) / float(G.N_PHASES)).unsqueeze(1)


def task_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Observation, the task rate r, shape (E, 1)."""
    return rate_buf(env).unsqueeze(1)


def sample_task_rate(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """Reset event, draws the task rate of the resetting environments from
    RATE_SPEC."""
    r = rate_buf(env)
    spec = RATE_SPEC
    n = len(env_ids)
    if spec["mode"] == "fixed":
        r[env_ids] = float(spec["value"])
    elif spec["mode"] == "uniform":
        lo, hi = float(spec["lo"]), float(spec["hi"])
        r[env_ids] = lo + (hi - lo) * torch.rand(n, device=env.device)
    elif spec["mode"] == "per_env":
        vals = torch.as_tensor(spec["values"], dtype=torch.float32, device=env.device)
        r[env_ids] = vals[env_ids]
    else:
        raise ValueError(spec)


def task_complete(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Termination, the cycle at the environment's rate has run out."""
    _, _, t, cycle = phase_state(env)
    return t >= cycle - 0.5 * float(env.step_dt)
