"""Pure-Python registry for the released ParcelStow benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TaskSpec:
    alias: str
    name: str
    gym_id: str
    driver: str
    default_rates: tuple[float, ...]
    default_output_dir: str
    default_checkpoint: str
    demonstration_artifact: str
    checkpoint_artifact: str
    stage_keys: tuple[str, ...]
    monitor: str
    expert: str
    phases: tuple[tuple[str, float, bool], ...]
    observation_pose_key: str
    observation_dim: int = 147
    action_dim: int = 16

    @property
    def driver_path(self) -> Path:
        return REPO / self.driver

    def cycle_time(self, rate: float) -> float:
        if rate <= 0:
            raise ValueError("speedup factor must be positive")
        return sum(duration / rate if scaled else duration for _, duration, scaled in self.phases)


_ACQUISITION = (("PARK", 0.5, False), ("APPROACH", 2.5, False),
                ("PREGRASP_DWELL", 0.6, False), ("CLOSE", 1.5, False),
                ("GRASP_DWELL", 0.6, False))

TASKS = {
    "parcel": TaskSpec(
        "parcel", "Parcel Insertion", "ParcelStow-L6-Distill-Play-v0",
        "scripts/manipulation/eval_stow_policies.py",
        (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5), "outputs/eval/parcel",
        "outputs/paper/act/act_stow.pt", "expert_demonstrations", "act_a_checkpoint",
        ("acquired", "lifted_clear", "reoriented", "preinsert_reached", "inserted", "released", "settled"),
        "parcelstow.tasks.manager_based.parcel_stow.mdp.metrics:StowMonitor",
        "scripts.manipulation.stow_runtime:ExpertActor",
        _ACQUISITION + (("LIFT", 1.2, True), ("REORIENT", 1.6, True),
                        ("TRANSFER", 1.6, True), ("PREINSERT_DWELL", 0.4, True),
                        ("INSERT", 1.0, True), ("INSERT_DWELL", 0.4, True),
                        ("RELEASE", 0.6, True), ("RETREAT", 1.0, True), ("SETTLE", 0.6, False)),
        "parcel_pose",
    ),
    "upright": TaskSpec(
        "upright", "Upright Placement", "UprightPlace-L6-Play-v0",
        "scripts/manipulation/eval_upright_policies.py",
        (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5), "outputs/eval/upright",
        "outputs/upright/act/act_upright.pt", "upright_expert_demonstrations", "upright_act_checkpoint",
        ("acquired", "lifted_clear", "reoriented_upright", "placed", "released", "settled"),
        "parcelstow.tasks.manager_based.upright_place.mdp.monitor:UprightMonitor",
        "scripts.manipulation.upright_runtime:UprightExpertActor",
        _ACQUISITION + (("LIFT", 2.4, True), ("REORIENT", 3.2, True),
                        ("TRANSFER", 3.2, True), ("LOWER", 2.0, True),
                        ("PLACE_DWELL", 0.8, True), ("RELEASE", 1.2, True),
                        ("RETREAT", 2.0, True), ("SETTLE", 1.0, False)),
        "object_pose",
    ),
    "peg": TaskSpec(
        "peg", "Keyed Peg Insertion", "PegInsert-L6-Play-v0",
        "scripts/manipulation/eval_peg_policies.py",
        (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5), "outputs/eval/peg",
        "outputs/peg/act/act_peg.pt", "peg_expert_demonstrations", "peg_act_checkpoint",
        ("acquired", "lifted_clear", "reoriented_upright", "aligned", "inserted", "released", "settled"),
        "parcelstow.tasks.manager_based.peg_insert.mdp.monitor:PegMonitor",
        "scripts.manipulation.peg_runtime:PegExpertActor",
        _ACQUISITION + (("LIFT", 2.4, True), ("REORIENT", 3.2, True),
                        ("TRANSFER", 3.2, True), ("INSERT", 3.0, True),
                        ("INSERT_DWELL", 0.8, True), ("RELEASE", 1.2, True),
                        ("RETREAT", 2.0, True), ("SETTLE", 2.0, False)),
        "object_pose",
    ),
}
ALIASES = tuple(TASKS)


def get_task(alias: str) -> TaskSpec:
    try:
        return TASKS[alias]
    except KeyError as exc:
        raise ValueError(f"unknown task {alias!r}; valid aliases: {', '.join(ALIASES)}") from exc


def task_output_dir(spec: TaskSpec, base: str | None, *, legacy_default: bool = False) -> str:
    if base:
        return base
    if legacy_default and spec.alias == "parcel":
        return "outputs/eval"
    return spec.default_output_dir
