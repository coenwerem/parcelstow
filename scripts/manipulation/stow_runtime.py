"""Shared runtime of the ParcelStow drivers, imported after the Isaac app
launches. Holds the actor wrappers (scripted expert, DAgger student,
Diffusion Policy, ACT), the batched episode runner with the physical
monitor, and the record helpers. Every driver (expert validation, rate
sweep, demonstration collection, DAgger, DP, ACT, evaluation, handoff,
video) goes through run_episodes so the episode protocol is one code path.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "manipulation"))

from parcelstow.tasks.manager_based.parcel_stow.parcel_stow_env_cfg import CHAIN_ACTUATED  # noqa: E402
from parcelstow.tasks.manager_based.parcel_stow import geometry as G  # noqa: E402
from parcelstow.tasks.manager_based.parcel_stow.mdp import task_clock  # noqa: E402
from parcelstow.tasks.manager_based.parcel_stow.mdp.metrics import StowMonitor, TRACE_COLUMNS  # noqa: E402

import parcel_stow_expert as pse  # noqa: E402

TASK = "ParcelStow-L6-Distill-Play-v0"


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    except Exception:
        return None


def config_stamp(base):
    """Frozen task configuration summary for every record."""
    gd = G.load_geometry_dict()
    return {
        "git_sha": git_sha(),
        "task": TASK,
        "parcel_extents": list(G.PARCEL_EXTENTS), "parcel_mass": G.PARCEL_MASS, "parcel_friction": G.PARCEL_FRICTION,
        "parcel_start": list(G.PARCEL_START),
        "receptacle": {"family": gd.get("family"), "entrance": gd.get("entrance"), "shelf_height": gd.get("shelf_height"),
                       "W_loose": gd.get("W_loose"), "W_tight": gd.get("W_tight"), "D_in": gd.get("D_in"),
                       "grasp_yaw": gd.get("grasp_yaw")},
        "phases": [[n, d, s] for n, d, s in G.PHASES],
        "control_dt": float(base.step_dt),
        "geometry_file": G.GEOMETRY_PATH,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ----------------------------------------------------------------------------
# actors
# ----------------------------------------------------------------------------
class ExpertActor:
    name = "expert"

    def __init__(self, base, bank=None, trajectory=None, candidate=0):
        self.base = base
        self.robot = base.scene["robot"]
        self.jids, self.jnames = self.robot.find_joints(CHAIN_ACTUATED, preserve_order=True)
        self.q_default = self.robot.data.default_joint_pos[:, self.jids]
        self.expert = pse.StowExpert(self.jnames, bank=bank, trajectory=trajectory, device=base.device,
                                     candidate=candidate)
        self.expert.allocate(base.num_envs)
        self.start_xy = torch.tensor(G.PARCEL_START[:2], device=base.device)

    def reset(self, ids, obs=None):
        ids = torch.as_tensor(list(ids), dtype=torch.long, device=self.base.device)
        if len(ids) == 0:
            return
        off = self.base._stow_start_pos[ids, :2] - self.start_xy
        self.expert.reset(ids, off)

    @torch.no_grad()
    def act(self, obs):
        k, f, _, _ = task_clock.phase_state(self.base)
        q_meas = self.robot.data.joint_pos[:, self.jids]
        return self.expert.act(k, f, self.q_default, q_meas)


class Student(nn.Module):
    """MLP student of scripts/distill/run_distill.py, standardized inputs."""

    def __init__(self, obs_dim, act_dim, obs_mean=None, obs_std=None):
        super().__init__()
        self.register_buffer("obs_mean", torch.zeros(obs_dim) if obs_mean is None else obs_mean.clone())
        self.register_buffer("obs_std", torch.ones(obs_dim) if obs_std is None else obs_std.clone())
        self.net = nn.Sequential(nn.Linear(obs_dim, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
                                 nn.Linear(256, 128), nn.ELU(), nn.Linear(128, act_dim))

    def forward(self, obs):
        return self.net((obs - self.obs_mean) / self.obs_std)


class DaggerActor:
    name = "dagger"

    def __init__(self, ckpt, device, n=None, model=None):
        if model is not None:
            self.model = model
        else:
            state = torch.load(ckpt, map_location=device)
            self.model = Student(state["obs_mean"].shape[0], state["net.6.bias"].shape[0]).to(device)
            self.model.load_state_dict(state)
        self.model.eval()

    def reset(self, ids, obs=None):
        pass

    @torch.no_grad()
    def act(self, obs):
        return self.model(obs), None


class DPActor:
    name = "dp"

    def __init__(self, ckpt, device, n):
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from third_party.diffusion_policy.conditional_unet1d import ConditionalUnet1D
        state = torch.load(ckpt, map_location=device)
        a = state["args"]
        self.To, self.T, self.Ta = a["n_obs_steps"], a["horizon"], a["n_action_steps"]
        self.steps = a["num_inference_steps"]
        self.obs_dim, self.act_dim = state["obs_dim"], state["act_dim"]
        self.model = ConditionalUnet1D(input_dim=self.act_dim, global_cond_dim=self.obs_dim * self.To,
                                       diffusion_step_embed_dim=256, down_dims=[256, 512, 1024],
                                       kernel_size=5, n_groups=8, cond_predict_scale=True).to(device)
        self.model.load_state_dict(state["ema"])
        self.model.eval()
        self.no_scale = state["norm_obs"]["scale"].to(device)
        self.no_off = state["norm_obs"]["offset"].to(device)
        self.na_scale = state["norm_act"]["scale"].to(device)
        self.na_off = state["norm_act"]["offset"].to(device)
        self.scheduler = DDPMScheduler(num_train_timesteps=100, beta_start=0.0001, beta_end=0.02,
                                       beta_schedule="squaredcos_cap_v2", variance_type="fixed_small",
                                       clip_sample=True, prediction_type="epsilon")
        self.device = device
        self.n = n
        self.hist = None
        self.queue = torch.zeros(n, self.Ta, self.act_dim, device=device)
        self.qpos = torch.full((n,), self.Ta, dtype=torch.long, device=device)

    def reset(self, ids, obs):
        if self.hist is None:
            self.hist = obs.unsqueeze(1).repeat(1, self.To, 1)
        for i in ids:
            self.hist[i] = obs[i].unsqueeze(0).repeat(self.To, 1)
            self.qpos[i] = self.Ta

    @torch.no_grad()
    def plan(self, obs_hist):
        nobs = obs_hist * self.no_scale + self.no_off
        cond = nobs.reshape(nobs.shape[0], -1)
        traj = torch.randn(nobs.shape[0], self.T, self.act_dim, device=self.device)
        self.scheduler.set_timesteps(self.steps)
        for t in self.scheduler.timesteps:
            out = self.model(traj, t, global_cond=cond)
            traj = self.scheduler.step(out, t, traj).prev_sample
        act = (traj - self.na_off) / self.na_scale
        s = self.To - 1
        return act[:, s:s + self.Ta]

    @torch.no_grad()
    def act(self, obs):
        self.hist = torch.cat([self.hist[:, 1:], obs.unsqueeze(1)], dim=1)
        need = self.qpos >= self.Ta
        if need.any():
            self.queue[need] = self.plan(self.hist[need])
            self.qpos[need] = 0
        a = self.queue[torch.arange(self.n, device=self.device), self.qpos]
        self.qpos += 1
        return a, None


class ACTActor:
    """State-only ACT with the released temporal ensembling (exponent
    0.01, query every step), on a ring buffer of the last chunk_size
    predictions so long episodes fit in memory."""

    name = "act"

    def __init__(self, ckpt, device, n):
        from state_act import StateACT
        state = torch.load(ckpt, map_location=device)
        a = state["args"]
        self.obs_mean = state["obs_mean"].to(device)
        self.obs_std = state["obs_std"].to(device)
        self.act_mean = state["act_mean"].to(device)
        self.act_std = state["act_std"].to(device)
        self.Tq = a["chunk_size"]
        self.act_dim = self.act_mean.shape[0]
        self.model = StateACT(self.obs_mean.shape[0], self.act_dim, self.Tq, hidden_dim=a["hidden_dim"],
                              dim_feedforward=a["dim_feedforward"]).to(device)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.temporal_agg = bool(a.get("temporal_agg", 1))
        self.k = 0.01
        self.n, self.device = n, device
        self.ring = torch.zeros(n, self.Tq, self.Tq, self.act_dim, device=device)
        self.tstep = torch.zeros(n, dtype=torch.long, device=device)
        self.ages = torch.arange(self.Tq, device=device)

    def reset(self, ids, obs=None):
        for i in ids:
            self.tstep[i] = 0
            self.ring[i] = 0.0

    @torch.no_grad()
    def act(self, obs):
        qpos = (obs - self.obs_mean) / self.obs_std
        a_hat, _, _ = self.model(qpos)
        a_hat = a_hat * self.act_std + self.act_mean
        if not self.temporal_agg:
            self.tstep += 1
            return a_hat[:, 0], None
        slot = self.tstep % self.Tq
        idx = torch.arange(self.n, device=self.device)
        self.ring[idx, slot] = a_hat
        # combine predictions of ages 0..Tq-1 covering the current step
        ages = self.ages
        src = (self.tstep.unsqueeze(1) - ages.unsqueeze(0))  # (n, Tq) step index of the prediction
        valid = src >= 0
        slots = torch.remainder(src, self.Tq)
        gathered = self.ring[idx.unsqueeze(1), slots, ages.unsqueeze(0)]  # (n, Tq, act_dim)
        # released weighting, exp(-k * rank) with rank 0 the OLDEST valid prediction
        n_valid = valid.sum(dim=1, keepdim=True)
        rank = (n_valid - 1 - ages.unsqueeze(0)).clamp(min=0).float()
        w = torch.exp(-self.k * rank) * valid.float()
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-9)
        acts = (gathered * w.unsqueeze(-1)).sum(dim=1)
        self.tstep += 1
        return acts, None


def load_actor(name, ckpt, base, n):
    device = base.device
    if name == "expert":
        return ExpertActor(base)
    if name == "dagger":
        return DaggerActor(ckpt, device, n)
    if name == "dp":
        return DPActor(ckpt, device, n)
    if name == "act":
        return ACTActor(ckpt, device, n)
    if ":" in name:
        # a user policy given as module.path:ClassName, see
        # docs/POLICY_INTERFACE.md and examples/custom_policy.py
        import importlib

        mod_name, cls_name = name.split(":", 1)
        cls = getattr(importlib.import_module(mod_name), cls_name)
        return cls(base=base, checkpoint=ckpt, num_envs=n)
    raise ValueError(name)


# ----------------------------------------------------------------------------
# environment switches
# ----------------------------------------------------------------------------
class EnvSwitches:
    def __init__(self, base, reset_term="reset_parcel"):
        self.base = base
        self.reset_parcel_cfg = base.event_manager.get_term_cfg(reset_term)
        self.obs_term_cfgs = base.observation_manager._group_obs_term_cfgs["policy"]
        self.saved_noise = [c.noise for c in self.obs_term_cfgs]

    def set_jitter(self, j):
        self.reset_parcel_cfg.params["pose_range"] = {"x": (-j, j), "y": (-j, j)} if j > 0 else {}

    def set_corruption(self, on):
        for c, nz in zip(self.obs_term_cfgs, self.saved_noise):
            c.noise = nz if on else None

    def set_rate(self, spec):
        task_clock.RATE_SPEC.clear()
        task_clock.RATE_SPEC.update(spec)


# ----------------------------------------------------------------------------
# episode runner
# ----------------------------------------------------------------------------
def run_episodes(env, base, actor, monitor, n_episodes, rate_spec, jitter, seed, switches,
                 expert=None, record_data=False, corrupt=False, action_noise=0.0, noise_half=False,
                 stamp=None, tag="", trace_dir=None, extra=None, verbose=True, max_steps=None,
                 step_hook=None, after_step_hook=None, record_hook=None,
                 task_id=None, cycle_time=None):
    """Roll the actor until n_episodes complete episodes are recorded.

    expert, an ExpertActor whose act() runs every step (its integrator
    follows the visited states) and whose action labels the data when
    record_data is set. Returns (records, episodes) where episodes is a
    list of (obs (T, D), expert_action (T, A), record) per recorded episode
    (empty unless record_data). Records hold policy, seed, task_rate,
    task_duration_s, and the monitor fields. Traces of monitor.trace_envs
    go to trace_dir as npz. step_hook(base, obs, act, monitor) may replace
    the action before env.step, after_step_hook(base, monitor, done) runs
    after the monitor update, and record_hook(env_index, record) may add
    fields to a finished episode's record before it is stored (the hooks
    of the handoff diagnostics).
    """
    n = base.num_envs
    device = base.device
    task_id = TASK if task_id is None else task_id
    cycle_time = G.cycle_time if cycle_time is None else cycle_time
    switches.set_rate(rate_spec)
    switches.set_jitter(jitter)
    switches.set_corruption(corrupt)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    all_ids = list(range(n))
    monitor.reset(all_ids)
    actor.reset(all_ids, obs)
    if expert is not None and expert is not actor:
        expert.reset(all_ids, obs)
    rate_at_reset = task_clock.rate_buf(base).clone()
    buf_obs = [[] for _ in range(n)]
    buf_act = [[] for _ in range(n)]
    ep_index = torch.zeros(n, dtype=torch.long, device=device)
    noisy = torch.arange(n, device=device) < (n // 2) if noise_half else torch.ones(n, dtype=torch.bool, device=device)
    records, episodes = [], []
    counter = 0
    t0 = time.time()
    steps = 0
    while len(records) < n_episodes:
        with torch.no_grad():
            a_exp, q_t_exp = (None, None)
            if expert is not None:
                a_exp, q_t_exp = expert.act(obs)
            if actor is expert:
                act, q_t = a_exp, q_t_exp
            else:
                act, q_t = actor.act(obs)
                if action_noise > 0:
                    act = act + action_noise * torch.randn_like(act) * noisy.unsqueeze(1).float()
                if q_t is None:
                    q_t = 0.5 * act + expert.q_default if expert is not None else None
            if record_data:
                for i in range(n):
                    buf_obs[i].append(obs[i].clone())
                    buf_act[i].append(a_exp[i].clone())
            if step_hook is not None:
                act = step_hook(base, obs, act, monitor)
            obs_dict, _, term, trunc, _ = env.step(act)
            obs = obs_dict["policy"]
            done = (term | trunc).bool().flatten()
            monitor.step(done, act, q_t)
            steps += 1
            if after_step_hook is not None:
                after_step_hook(base, monitor, done)
            done_ids = done.nonzero(as_tuple=False).flatten().tolist()
            for i in done_ids:
                if len(records) >= n_episodes:
                    break
                rec = monitor.episode_record(i)
                r = float(rate_at_reset[i])
                rec.update({
                    "task": task_id,
                    "policy": actor.name, "tag": tag, "seed": int(seed), "episode": counter,
                    "env": i, "task_rate": r, "task_duration_s": cycle_time(r),
                    "jitter": jitter, "corrupt": bool(corrupt), "action_noise": action_noise,
                })
                if stamp is not None:
                    rec["config"] = stamp
                if extra:
                    rec.update(extra)
                if record_hook is not None:
                    record_hook(i, rec)
                records.append(rec)
                counter += 1
                if record_data:
                    episodes.append((torch.stack(buf_obs[i]).cpu(), torch.stack(buf_act[i]).cpu(), rec))
                if trace_dir is not None and i in monitor.trace_envs:
                    arr = monitor.take_trace(i)
                    if arr is not None:
                        os.makedirs(trace_dir, exist_ok=True)
                        np.savez_compressed(os.path.join(trace_dir, f"{tag}_ep{counter - 1:04d}.npz"),
                                            trace=arr, columns=np.array(TRACE_COLUMNS), record=json.dumps(rec))
                if verbose and (counter % 10 == 0 or counter == n_episodes):
                    ok = sum(1 for x in records if x["task_success"])
                    print(f"[{tag}] {counter}/{n_episodes} episodes, success {ok}/{counter}, "
                          f"{steps} steps, {time.time() - t0:.0f}s", flush=True)
            for i in done_ids:
                buf_obs[i] = []
                buf_act[i] = []
            if done_ids:
                monitor.reset(done_ids)
                actor.reset(done_ids, obs)
                if expert is not None and expert is not actor:
                    expert.reset(done_ids, obs)
                rate_at_reset[done_ids] = task_clock.rate_buf(base)[done_ids]
            if max_steps is not None and steps >= max_steps:
                break
    return records, episodes


V1_STAGE_KEYS = ["acquired", "lifted_clear", "reoriented", "preinsert_reached",
                 "inserted", "released", "settled"]


def summarize(records, stage_keys=None):
    """Success and stage rates with Wilson intervals over a record list.
    stage_keys defaults to the ParcelStow stages; a second task passes its
    own stage names."""
    n = len(records)
    if n == 0:
        return {"episodes": 0}
    stage_keys = V1_STAGE_KEYS if stage_keys is None else list(stage_keys)
    def frac(key):
        k = sum(1 for r in records if r.get(key))
        lo, hi = G.wilson(k, n)
        return {"frac": k / n, "k": k, "n": n, "wilson": [lo, hi]}
    reasons = {}
    for r in records:
        reasons[r["failure_reason"]] = reasons.get(r["failure_reason"], 0) + 1
    def dist(key):
        v = np.array([r[key] for r in records if r.get(key) is not None and np.isfinite(r[key])], dtype=float)
        if v.size == 0:
            return None
        return {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "max": float(v.max()),
                "mean": float(v.mean())}
    out = {
        "episodes": n,
        "task_success": frac("task_success"),
    }
    for key in stage_keys:
        out[key] = frac(key)
    out.update({
        "failure_reasons": reasons,
        "max_hand_object_translation_m": dist("max_hand_object_translation_m"),
        "max_hand_object_rotation_deg": dist("max_hand_object_rotation_deg"),
        "peak_hand_linear_velocity": dist("peak_hand_linear_velocity"),
        "peak_hand_angular_velocity": dist("peak_hand_angular_velocity"),
        "max_joint_velocity_utilization": dist("max_joint_velocity_utilization"),
        "max_arm_velocity_utilization": dist("max_arm_velocity_utilization"),
        "max_target_tracking_error_rad": dist("max_target_tracking_error_rad"),
        "max_receptacle_force": dist("max_receptacle_force"),
        "epsilon_lift": dist("epsilon_lift"), "epsilon_beta_lift": dist("epsilon_beta_lift"),
    })
    if "task" in records[0]:
        out["task"] = records[0]["task"]
    return out


def write_jsonl(path, rows, mode="a"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, mode) as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def light_record(rec):
    """Record without the contact sets (for compact summaries)."""
    return {k: v for k, v in rec.items() if not k.startswith("contact_set_")}
