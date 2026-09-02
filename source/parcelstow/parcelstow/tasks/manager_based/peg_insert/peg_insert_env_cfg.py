"""Keyed peg insertion task, reorienting a cuboid and inserting it into a
square pocket with 3 mm of clearance per side.

Fixed-pelvis G1 with the RealHand L6, the v1 tabletop, the shared
55 x 55 x 180 mm cuboid lying on its side, and a pocket block of nine
kinematic slabs (floor, four walls, four lead-in slabs) on the robot's
right of the transport axis, the v1 receptacle pattern. The control interface is the v1 one; the policy observes the
147-D state vector with the peg pose in the object slice. Task success
is decided by the physical monitor (mdp/monitor.py) in the drivers.

The peg is a free rigid body. The environment binds the peg phase
schedule to the shared task clock at configuration instantiation, one
task per process.
"""

import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from parcelstow.phase_schedule import PhaseSchedule
from parcelstow.robots import G1_L6_CFG

from ..parcel_stow import agents
from ..parcel_stow.mdp import task_clock
from ..parcel_stow.parcel_stow_env_cfg import CHAIN_ACTUATED, TABLE_POS, TABLE_SIZE  # noqa: F401
from . import geometry as P
from . import mdp

EPISODE_LENGTH_S = 45.0  # covers the 39.3 s cycle at r = 0.5
START_QUAT = tuple(float(v) for v in P.quat_from_mat(P.R_START))
SLAB_NAMES = ["floor", "wall_a", "wall_b", "wall_c", "wall_d",
              "lead_a", "lead_b", "lead_c", "lead_d"]
SLAB_PRIMS = {n: f"Pocket_{n}" for n in SLAB_NAMES}
_slabs = P.pocket_slabs()


def _slab_cfg(name: str) -> RigidObjectCfg:
    """One pocket slab, a kinematic rigid body (the v1 receptacle
    pattern, so the peg-filtered contact sensor resolves it)."""
    slab = _slabs[name]
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + SLAB_PRIMS[name],
        spawn=sim_utils.CuboidCfg(
            size=tuple(float(v) for v in slab["size"]),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.50)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(float(v) for v in slab["center"]),
                                                  rot=tuple(float(v) for v in slab["quat_wxyz"])),
    )


@configclass
class PegInsertSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    robot = G1_L6_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.5, 0.38)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POS),
    )
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=P.OBJECT_EXTENTS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=P.OBJECT_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=P.OBJECT_FRICTION, dynamic_friction=P.OBJECT_FRICTION, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.30, 0.55)),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=P.START_POS, rot=START_QUAT),
    )
    # Visual pedestal completing the table-mounted block below the
    # functional slabs (the kinematic shell starts at the floor slab,
    # 50 mm above the table, and reads as floating on camera). No
    # collider: the slabs carry all the physics and nothing reaches
    # under the block, so validation is untouched.
    pocket_pedestal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pocket_pedestal",
        spawn=sim_utils.CuboidCfg(
            size=(P.POCKET_W + 2 * P.WALL_T, P.POCKET_W + 2 * P.WALL_T,
                  P.POCKET_FLOOR_Z - P.FLOOR_T - P.TABLE_TOP),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.50)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(P.POCKET_CENTER[0], P.POCKET_CENTER[1],
                 (P.TABLE_TOP + P.POCKET_FLOOR_Z - P.FLOOR_T) / 2),
            rot=tuple(float(v) for v in P.quat_from_mat(P.R_POCKET))),
    )
    pocket_floor = _slab_cfg("floor")
    pocket_wall_a = _slab_cfg("wall_a")
    pocket_wall_b = _slab_cfg("wall_b")
    pocket_wall_c = _slab_cfg("wall_c")
    pocket_wall_d = _slab_cfg("wall_d")
    pocket_lead_a = _slab_cfg("lead_a")
    pocket_lead_b = _slab_cfg("lead_b")
    pocket_lead_c = _slab_cfg("lead_c")
    pocket_lead_d = _slab_cfg("lead_d")
    # net contact forces of the five distal phalanges (observation)
    tip_contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/rh_.*_distal", history_length=1)
    # object-filtered per-body forces (realized contact set, slip diagnostics)
    rh_thumb_distal_object_s = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rh_thumb_distal", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"])
    rh_index_distal_object_s = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rh_index_distal", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"])
    rh_middle_distal_object_s = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rh_middle_distal", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"])
    rh_ring_distal_object_s = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rh_ring_distal", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"])
    rh_pinky_distal_object_s = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rh_pinky_distal", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"])
    # peg against the pocket slabs (jam diagnostics)
    peg_pocket_s = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/" + SLAB_PRIMS[n] for n in SLAB_NAMES])


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=CHAIN_ACTUATED,
        preserve_order=True,
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), clip=(-10.0, 10.0))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5), clip=(-50.0, 50.0))
        actions = ObsTerm(func=mdp.last_action, clip=(-10.0, 10.0))
        object_pose = ObsTerm(func=mdp.object_pose_b, noise=Unoise(n_min=-0.01, n_max=0.01), clip=(-5.0, 5.0))
        fingertip_pos = ObsTerm(func=mdp.fingertip_pos_b, noise=Unoise(n_min=-0.005, n_max=0.005))
        tip_forces = ObsTerm(func=mdp.tip_contact_forces)
        task_phase = ObsTerm(func=mdp.task_phase)
        task_rate = ObsTerm(func=mdp.task_rate)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        },
    )
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("object")},
    )
    record_object_start = EventTerm(func=mdp.record_parcel_start, mode="reset",
                                    params={"parcel_cfg": SceneEntityCfg("object")})
    sample_task_rate = EventTerm(func=mdp.sample_task_rate, mode="reset")


@configclass
class RewardsCfg:
    # No learning reward, one zero-weight term satisfies the manager.
    alive = RewTerm(func=mdp.is_alive, weight=0.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    task_complete = DoneTerm(func=mdp.task_complete, time_out=True)
    object_fell = DoneTerm(func=mdp.parcel_fell, params={"parcel_cfg": SceneEntityCfg("object")})
    exploding_state = DoneTerm(func=mdp.exploding_state, params={"max_joint_vel": 100.0})


@configclass
class PegInsertEnvCfg(ManagerBasedRLEnvCfg):
    scene: PegInsertSceneCfg = PegInsertSceneCfg(num_envs=50, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = EPISODE_LENGTH_S
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.scene.robot.spawn.articulation_props.fix_root_link = True
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.75)
        # The left arm stays at the v1 default (the upright lesson: a
        # static re-park breaks the open-loop acquisition through torso
        # sag); the pocket top at 0.770 m sits above the idle-hand zone.
        self.scene.robot.actuators["arms"].stiffness = 300.0
        self.scene.robot.actuators["arms"].damping = 10.0
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 16
        self.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 8
        self.viewer.eye = (1.6, -1.4, 1.35)
        self.viewer.lookat = (0.45, -0.1, 0.8)
        # One task per process: bind the peg schedule to the shared clock.
        task_clock.SCHEDULE = PhaseSchedule(P.PHASES)


@configclass
class PegInsertEnvCfg_PLAY(PegInsertEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False


gym.register(
    id="PegInsert-L6-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:PegInsertEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ParcelStowPPORunnerCfg",
    },
)
