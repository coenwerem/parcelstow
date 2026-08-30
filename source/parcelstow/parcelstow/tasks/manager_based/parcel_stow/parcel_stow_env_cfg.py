"""ParcelStow task, dexterous parcel reorientation and stowing
(docs/TASK_SPEC.md).

Fixed-pelvis G1 with the LinkerHand L6, the tabletop, a rigid 80 x 55 x 40
mm parcel, and an open-front receptacle built from five static collision
boxes. The control interface is the one of the cube tasks, absolute
joint-position targets on the 16 joints of CHAIN_ACTUATED at 50 Hz with the
same PD gains. The policy observes joint state, its last action, the parcel
pose in the pelvis frame, the distal phalanx positions and contact forces,
the task phase, and the speedup factor r. No GDF, bank, or force-closure
quantity enters the observation, the rewards, or the terminations. Task success is decided by
the physical monitor (mdp/metrics.py) in the drivers, never by the reward.

The parcel is a free rigid body. Nothing attaches it to the hand, and the
receptacle slabs are plain static colliders. The geometry constants come
from assets/parcel_stow_geometry.json, written by
scripts/manipulation/probe_stow_geometry.py --finalize.
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

from parcelstow.robots import G1_L6_CFG

from . import agents
from . import geometry as G
from . import mdp

# The 16 actuated joints of the control chain, waist plus right arm plus
# five finger drives, in the action order of the 16-D joint-position target.
CHAIN_ACTUATED = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "rh_thumb_cmc_roll",
    "rh_thumb_cmc_pitch",
    "rh_index_mcp_pitch",
    "rh_middle_mcp_pitch",
    "rh_ring_mcp_pitch",
    "rh_pinky_mcp_pitch",
]

# Work-surface placement of the frozen scene.
TABLE_POS = (0.55, 0.0, 0.68)
TABLE_SIZE = (0.81, 1.092, 0.04)

DISTAL_BODIES = ["rh_thumb_distal", "rh_index_distal", "rh_middle_distal", "rh_ring_distal", "rh_pinky_distal"]
SLAB_NAMES = ["floor", "side_a", "side_b", "back", "top"]
SLAB_PRIMS = {n: f"Cubby_{n}" for n in SLAB_NAMES}
EPISODE_LENGTH_S = 30.0

_geom_dict = G.load_geometry_dict()
GEOM = G.StowGeometry.from_dict(_geom_dict)


def _slab_cfg(name: str) -> RigidObjectCfg:
    """One receptacle slab, a kinematic rigid body (it never moves, and the
    rigid-body API lets the parcel-filtered contact sensor resolve it)."""
    slab = _geom_dict["slabs"][name]
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
class ParcelStowSceneCfg(InteractiveSceneCfg):
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
    parcel = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Parcel",
        spawn=sim_utils.CuboidCfg(
            size=G.PARCEL_EXTENTS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=G.PARCEL_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=G.PARCEL_FRICTION, dynamic_friction=G.PARCEL_FRICTION, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.35, 0.10)),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=G.PARCEL_START, rot=tuple(float(v) for v in _geom_dict["start_quat_wxyz"])),
    )
    cubby_floor = _slab_cfg("floor")
    cubby_side_a = _slab_cfg("side_a")
    cubby_side_b = _slab_cfg("side_b")
    cubby_back = _slab_cfg("back")
    cubby_top = _slab_cfg("top")
    # net contact forces of the five distal phalanges (observation)
    tip_contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/rh_.*_distal", history_length=1)
    # parcel-filtered per-body forces (realized contact set, slip diagnostics)
    rh_thumb_distal_parcel_s = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rh_thumb_distal", filter_prim_paths_expr=["{ENV_REGEX_NS}/Parcel"])
    rh_index_distal_parcel_s = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rh_index_distal", filter_prim_paths_expr=["{ENV_REGEX_NS}/Parcel"])
    rh_middle_distal_parcel_s = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rh_middle_distal", filter_prim_paths_expr=["{ENV_REGEX_NS}/Parcel"])
    rh_ring_distal_parcel_s = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rh_ring_distal", filter_prim_paths_expr=["{ENV_REGEX_NS}/Parcel"])
    rh_pinky_distal_parcel_s = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rh_pinky_distal", filter_prim_paths_expr=["{ENV_REGEX_NS}/Parcel"])
    # parcel against the receptacle slabs (insertion contact diagnostics)
    parcel_cubby_s = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Parcel",
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
        parcel_pose = ObsTerm(func=mdp.parcel_pose_b, noise=Unoise(n_min=-0.01, n_max=0.01), clip=(-5.0, 5.0))
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
    reset_parcel = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("parcel")},
    )
    record_parcel_start = EventTerm(func=mdp.record_parcel_start, mode="reset")
    sample_task_rate = EventTerm(func=mdp.sample_task_rate, mode="reset")


@configclass
class RewardsCfg:
    # The task has no learning reward. One zero-weight term satisfies the
    # manager, imitation drives every learner.
    alive = RewTerm(func=mdp.is_alive, weight=0.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    task_complete = DoneTerm(func=mdp.task_complete, time_out=True)
    parcel_fell = DoneTerm(func=mdp.parcel_fell)
    exploding_state = DoneTerm(func=mdp.exploding_state, params={"max_joint_vel": 100.0})


@configclass
class ParcelStowEnvCfg(ManagerBasedRLEnvCfg):
    scene: ParcelStowSceneCfg = ParcelStowSceneCfg(num_envs=50, env_spacing=4.0)
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
        self.scene.robot.init_state.pos = G.PELVIS_POS
        self.scene.robot.actuators["arms"].stiffness = 300.0
        self.scene.robot.actuators["arms"].damping = 10.0
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 16
        self.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 8
        self.viewer.eye = (1.6, -1.4, 1.35)
        self.viewer.lookat = (0.45, -0.1, 0.8)


@configclass
class ParcelStowEnvCfg_PLAY(ParcelStowEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False


gym.register(
    id="ParcelStow-L6-Distill-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:ParcelStowEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ParcelStowPPORunnerCfg",
    },
)
