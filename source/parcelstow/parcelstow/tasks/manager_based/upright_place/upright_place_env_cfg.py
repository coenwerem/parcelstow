"""Upright placement task, reorienting a tall cuboid to an upright pose on a
marked target region.

Fixed-pelvis G1 with the RealHand L6, the v1 tabletop, a rigid
55 x 55 x 180 mm cuboid lying on its side, and a visual target disk on
the table. The control interface is the v1 one, absolute joint-position
targets on the 16 joints of CHAIN_ACTUATED at 50 Hz with the same PD
gains. The policy observes joint state, its last action, the object
pose in the pelvis frame, the distal phalanx positions and contact
forces, the task phase, and the speedup factor r. Task success is
decided by the physical monitor (mdp/monitor.py) in the drivers, never
by the reward.

The object is a free rigid body; nothing attaches it to the hand. The
environment binds the upright phase schedule to the shared task clock
at configuration instantiation, so one process runs one task, the
Isaac constraint the v1 test fixture already documents.
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
from . import geometry as U
from . import mdp

EPISODE_LENGTH_S = 45.0  # covers the 36.3 s cycle at r = 0.5 under the calibrated schedule
START_QUAT = tuple(float(v) for v in U.quat_from_mat(U.R_START))


@configclass
class UprightPlaceSceneCfg(InteractiveSceneCfg):
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
            size=U.OBJECT_EXTENTS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=U.OBJECT_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=U.OBJECT_FRICTION, dynamic_friction=U.OBJECT_FRICTION, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.45, 0.75)),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=U.START_POS, rot=START_QUAT),
    )
    # The target region is defined by the geometry constants alone; no
    # collider enters the scene, so nothing can perturb the placed
    # object at the target. The disk below is visual only, the marked
    # region the task docstring names.
    target_disk = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetDisk",
        spawn=sim_utils.CylinderCfg(
            radius=U.TARGET_RADIUS, height=0.001,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.62, 0.42)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(U.TARGET_CENTER[0], U.TARGET_CENTER[1], TABLE_POS[2] + TABLE_SIZE[2] / 2 + 0.001)),
    )
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
class UprightPlaceEnvCfg(ManagerBasedRLEnvCfg):
    scene: UprightPlaceSceneCfg = UprightPlaceSceneCfg(num_envs=50, env_spacing=4.0)
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
        # The left arm stays at the v1 default: any static re-park shifts
        # the waist gravity load and the resulting torso sag enough to
        # break the millimeter-margin open-loop acquisition (measured, two
        # park variants). The placement target instead sits on the robot's
        # right of the transport axis, outside the idle left hand's zone;
        # the schedule test asserts the clearance.
        self.scene.robot.actuators["arms"].stiffness = 300.0
        self.scene.robot.actuators["arms"].damping = 10.0
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 16
        self.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 8
        self.viewer.eye = (1.6, -1.4, 1.35)
        self.viewer.lookat = (0.45, -0.1, 0.8)
        # One task per process: bind the upright schedule to the shared clock.
        task_clock.SCHEDULE = PhaseSchedule(U.PHASES)


@configclass
class UprightPlaceEnvCfg_PLAY(UprightPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False


gym.register(
    id="UprightPlace-L6-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:UprightPlaceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ParcelStowPPORunnerCfg",
    },
)
