"""Record videos of upright placement and keyed peg insertion episodes, the
record_stow_rollouts.py camera pattern over the upright placement and
keyed-peg insertion tasks.

One environment, a fixed camera over the robot's right shoulder that
frames the right hand, the object, the table, and the task fixture
(the target region or the pocket block). Every control step renders one
frame and the video plays at 25 fps from every second frame, real time
at r = 1. A JSON sidecar stores the episode record, the camera pose,
and the configuration stamp.

Run,
  python scripts/manipulation/record_task_rollouts.py --task peg \
      --actor expert --rate 1.0 --episodes 1 --select success --tag peg_expert_r1
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

TASKS = {
    "upright": {
        "id": "UprightPlace-L6-Play-v0",
        "ckpt": "outputs/upright/act/act_upright.pt",
        "eye": [0.28, -0.80, 1.30],
        "target": [0.46, 0.03, 0.86],
        "out_dir": "outputs/upright/videos",
    },
    "peg": {
        "id": "PegInsert-L6-Play-v0",
        "ckpt": "outputs/peg/act/act_peg.pt",
        "eye": [0.30, -0.85, 1.32],
        "target": [0.50, 0.08, 0.86],
        "out_dir": "outputs/peg/videos",
    },
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=list(TASKS), required=True)
parser.add_argument("--actor", type=str, default="expert")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--rate", type=float, default=1.0)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--jitter", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--select", type=str, default="first", choices=["first", "success", "failure"])
parser.add_argument("--max_episodes", type=int, default=30)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--eye", type=float, nargs=3, default=None)
parser.add_argument("--target", type=float, nargs=3, default=None)
parser.add_argument("--out_dir", type=str, default=None)
parser.add_argument("--tag", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True
spec = TASKS[args_cli.task]
for name in ("eye", "target", "out_dir"):
    if getattr(args_cli, name) is None:
        setattr(args_cli, name, spec[name])
if args_cli.checkpoint is None:
    args_cli.checkpoint = spec["ckpt"]
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import parcelstow.tasks  # noqa: E402, F401
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stow_runtime as rt  # noqa: E402

if args_cli.task == "peg":
    from parcelstow.tasks.manager_based.peg_insert.mdp.monitor import PegMonitor as Monitor  # noqa: E402
    from peg_runtime import SCHED, config_stamp  # noqa: E402
    from peg_runtime import PegExpertActor as ExpertActor
else:
    from parcelstow.tasks.manager_based.upright_place.mdp.monitor import UprightMonitor as Monitor  # noqa: E402
    from upright_runtime import SCHED, config_stamp  # noqa: E402
    from upright_runtime import UprightExpertActor as ExpertActor


@hydra_task_config(spec["id"], "rsl_rl_cfg_entry_point")
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
    env = gym.make(spec["id"], cfg=env_cfg)
    base = env.unwrapped
    camera = base.scene["camera"]
    eye = torch.tensor([args_cli.eye], dtype=torch.float32, device=base.device) + base.scene.env_origins
    tgt = torch.tensor([args_cli.target], dtype=torch.float32, device=base.device) + base.scene.env_origins
    camera.set_world_poses_from_view(eye, tgt)
    monitor = Monitor(base)
    expert = ExpertActor(base)
    switches = rt.EnvSwitches(base, reset_term="reset_object")
    stamp = config_stamp(base, task_id=spec["id"])
    actor = expert if args_cli.actor == "expert" else rt.load_actor(args_cli.actor, args_cli.checkpoint, base, 1)
    os.makedirs(args_cli.out_dir, exist_ok=True)
    frames = []

    def after_step(base_, mon, done):
        img = camera.data.output["rgb"][0]
        if img.shape[-1] == 4:
            img = img[..., :3]
        frames.append(img.detach().cpu().numpy().astype(np.uint8))

    def flush(rec, index):
        if not frames:
            return
        vid = os.path.join(args_cli.out_dir, f"{args_cli.tag}_ep{index:02d}.mp4")
        with imageio.get_writer(vid, fps=25, codec="libx264", quality=8, macro_block_size=8) as wr:
            for i in range(0, len(frames), 2):
                wr.append_data(frames[i])
        meta = {"tag": args_cli.tag, "task": spec["id"], "actor": args_cli.actor, "rate": args_cli.rate,
                "checkpoint": args_cli.checkpoint if args_cli.actor != "expert" else None,
                "video": vid, "frames": len(frames), "record": rt.light_record(rec), "config": stamp,
                "camera": {"eye": args_cli.eye, "target": args_cli.target}}
        with open(vid.replace(".mp4", ".json"), "w") as fh:
            json.dump(meta, fh)
        print(f"[record] wrote {vid} ({len(frames)} frames), success {rec['task_success']} "
              f"reason {rec['failure_reason']}", flush=True)

    recorded = 0
    tried = 0
    while recorded < args_cli.episodes and tried < args_cli.max_episodes:
        frames.clear()
        recs, _ = rt.run_episodes(env, base, actor, monitor, 1, {"mode": "fixed", "value": args_cli.rate},
                                  args_cli.jitter, args_cli.seed + tried, switches, expert=expert, corrupt=False,
                                  stamp=stamp, tag=args_cli.tag, after_step_hook=after_step, verbose=False,
                                  task_id=spec["id"], cycle_time=SCHED.cycle_time)
        rec = recs[0]
        tried += 1
        want = (args_cli.select == "first" or (args_cli.select == "success" and rec["task_success"])
                or (args_cli.select == "failure" and not rec["task_success"]))
        if want:
            flush(rec, recorded)
            recorded += 1
        else:
            print(f"[record] skipped episode {tried} (success {rec['task_success']}, want {args_cli.select})",
                  flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
