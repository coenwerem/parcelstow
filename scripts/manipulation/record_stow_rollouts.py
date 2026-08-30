"""Record videos and deterministic stills of ParcelStow episodes (M13).

One environment, a fixed camera over the robot's right shoulder that frames
the right hand, the parcel, the table, the reorientation, the receptacle,
and the insertion. Every control step renders one frame (video at 25 fps
from every second frame), and stills are taken at the physical events the
monitor reports, initial, grasp established (acquired), lift (parcel 60 mm
up), about 45 deg of reorientation, about 90 deg (reoriented), pre-insertion
reached, insertion contact (first receptacle force or inserted), and
release/settled (or the episode end). Frames have no overlay, the montage
script draws the minimal diagnostic text (actor, speedup factor r,
epsilon^(beta) at lift, max in-hand slip, task outcome) from the saved
episode record.

Run,
  python scripts/manipulation/record_stow_rollouts.py --actor expert --rate 1.0 --episodes 1 \
      --out_dir outputs/paper/videos --tag expert_nominal --enable_cameras
"""

import argparse
import json
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="ParcelStow-L6-Distill-Play-v0")
parser.add_argument("--actor", type=str, default="expert", choices=["expert", "dagger", "dp", "act"])
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--rate", type=float, default=1.0)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--jitter", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--select", type=str, default="first", choices=["first", "success", "failure"],
                    help="record the first episode, or the first success, or the first failure (up to --max_episodes)")
parser.add_argument("--max_episodes", type=int, default=30)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--eye", type=float, nargs=3, default=[0.22, -0.78, 1.28])
parser.add_argument("--target", type=float, nargs=3, default=[0.46, 0.12, 0.85])
parser.add_argument("--out_dir", type=str, default="outputs/paper/videos")
parser.add_argument("--stills_dir", type=str, default="outputs/paper/stills")
parser.add_argument("--tag", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import parcelstow.tasks  # noqa: E402, F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stow_runtime as rt  # noqa: E402
from parcelstow.tasks.manager_based.parcel_stow import geometry as G  # noqa: E402
from parcelstow.tasks.manager_based.parcel_stow.mdp import task_clock  # noqa: E402
from parcelstow.tasks.manager_based.parcel_stow.mdp.metrics import StowMonitor  # noqa: E402

STILL_EVENTS = ["initial", "grasp_established", "lift", "reorient_45", "reorient_90", "preinsert",
                "insertion_contact", "release_settled"]


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 1
    env_cfg.scene.camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,
        height=args_cli.height, width=args_cli.width, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=1.2, horizontal_aperture=20.955,
                                         clipping_range=(0.05, 20.0)),
    )
    env_cfg.sim.render_interval = env_cfg.decimation
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    camera = base.scene["camera"]
    eye = torch.tensor([args_cli.eye], dtype=torch.float32, device=base.device) + base.scene.env_origins
    tgt = torch.tensor([args_cli.target], dtype=torch.float32, device=base.device) + base.scene.env_origins
    camera.set_world_poses_from_view(eye, tgt)
    geom = G.load_geometry()
    monitor = StowMonitor(base, geom)
    expert = rt.ExpertActor(base)
    switches = rt.EnvSwitches(base)
    stamp = rt.config_stamp(base)
    actor = expert if args_cli.actor == "expert" else rt.load_actor(args_cli.actor, args_cli.checkpoint, base, 1)
    os.makedirs(args_cli.out_dir, exist_ok=True)
    os.makedirs(args_cli.stills_dir, exist_ok=True)

    state = {"frames": [], "stills": {}, "meta": [], "ep": 0, "R_start": torch.tensor(G.quat_from_mat(geom.R_start),
             dtype=torch.float32, device=base.device).unsqueeze(0)}
    from isaaclab.utils.math import quat_mul, quat_inv

    def grab():
        img = camera.data.output["rgb"][0]
        if img.shape[-1] == 4:
            img = img[..., :3]
        return img.detach().cpu().numpy().astype(np.uint8)

    def after_step(base_, mon, done):
        img = grab()
        state["frames"].append(img)
        k, f, t, _ = task_clock.phase_state(base_)
        parcel = base_.scene["parcel"]
        q = parcel.data.root_quat_w
        w = quat_mul(quat_inv(state["R_start"]), q)[:, 0].abs().clamp(max=1.0)
        ang_from_start = math.degrees(2.0 * math.acos(float(w[0])))
        z_rel = float(parcel.data.root_pos_w[0, 2] - base_.scene.env_origins[0, 2] - base_._stow_start_pos[0, 2])
        st = state["stills"]
        def mark(name):
            if name not in st:
                st[name] = (len(state["frames"]) - 1, img)
        if len(state["frames"]) == 1:
            mark("initial")
        if bool(mon.acquired[0]):
            mark("grasp_established")
        if bool(mon.acquired[0]) and z_rel >= 0.06:
            mark("lift")
        if bool(mon.acquired[0]) and ang_from_start >= 45.0:
            mark("reorient_45")
        if bool(mon.reoriented[0]):
            mark("reorient_90")
        if bool(mon.preinsert_reached[0]):
            mark("preinsert")
        if float(mon.max_cubby_force[0]) > 0.0 or bool(mon.inserted[0]):
            mark("insertion_contact")
        if bool(mon.settled[0]) or bool(mon.released[0]):
            mark("release_settled")
        state["meta"].append({"t": float(t[0]), "k": int(k[0]), "f": float(f[0]), "ang_from_start_deg": ang_from_start,
                              "slip_t_m": float(mon.max_slip_t[0]), "slip_r_deg": math.degrees(float(mon.max_slip_r[0]))})

    def flush(rec, index):
        frames = state["frames"]
        if not frames:
            return
        vid = os.path.join(args_cli.out_dir, f"{args_cli.tag}_ep{index:02d}.mp4")
        with imageio.get_writer(vid, fps=25, codec="libx264", quality=8, macro_block_size=8) as wr:
            for i in range(0, len(frames), 2):
                wr.append_data(frames[i])
        stills = state["stills"]
        if "release_settled" not in stills:
            stills["release_settled"] = (len(frames) - 1, frames[-1])
        for name, (fi, img) in stills.items():
            imageio.imwrite(os.path.join(args_cli.stills_dir, f"{args_cli.tag}_ep{index:02d}_{name}.png"), img)
        meta = {"tag": args_cli.tag, "actor": args_cli.actor, "rate": args_cli.rate, "checkpoint": args_cli.checkpoint,
                "video": vid, "frames": len(frames), "still_frames": {n: fi for n, (fi, _) in stills.items()},
                "record": rt.light_record(rec), "trace": state["meta"], "config": stamp,
                "camera": {"eye": args_cli.eye, "target": args_cli.target}}
        with open(os.path.join(args_cli.stills_dir, f"{args_cli.tag}_ep{index:02d}.json"), "w") as fh:
            json.dump(meta, fh)
        print(f"[record] wrote {vid} ({len(frames)} frames), stills {sorted(stills.keys())}, "
              f"success {rec['task_success']} reason {rec['failure_reason']}", flush=True)

    recorded = 0
    tried = 0
    while recorded < args_cli.episodes and tried < args_cli.max_episodes:
        state["frames"], state["stills"], state["meta"] = [], {}, []
        recs, _ = rt.run_episodes(env, base, actor, monitor, 1, {"mode": "fixed", "value": args_cli.rate},
                                  args_cli.jitter, args_cli.seed + tried, switches, expert=expert, corrupt=False,
                                  stamp=stamp, tag=f"{args_cli.tag}", after_step_hook=after_step, verbose=False)
        rec = recs[0]
        tried += 1
        want = (args_cli.select == "first" or (args_cli.select == "success" and rec["task_success"])
                or (args_cli.select == "failure" and not rec["task_success"]))
        if want:
            flush(rec, recorded)
            recorded += 1
        else:
            print(f"[record] skipped episode {tried} (success {rec['task_success']}, want {args_cli.select})", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
