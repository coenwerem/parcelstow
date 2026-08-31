"""Articulation configs for the G1 with the LinkerHand L6 right hand.

G1_L6_CFG copies the provided G1_29DOF_CFG actuator layout and swaps the
asset for the locally merged URDF conversion. The six actuated hand joints
get grasp-oriented gains following the provided Inspire-hand config. The
five dip joints run at zero stiffness because the PhysxMimicJointAPI
constraint drives them from their MCP or CMC reference joints.
"""

import os

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from isaaclab_assets.robots.unitree import G1_29DOF_CFG

_ASSET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets")
)

G1_L6_USD_PATH = os.path.join(_ASSET_DIR, "g1_l6", "usd_both", "g1_29dof_l6_both.usd")

G1_L6_CFG = G1_29DOF_CFG.copy()
G1_L6_CFG.spawn.usd_path = G1_L6_USD_PATH
G1_L6_CFG.spawn.activate_contact_sensors = True

del G1_L6_CFG.actuators["hands"]

G1_L6_CFG.actuators["l6_hand"] = ImplicitActuatorCfg(
    joint_names_expr=[
        "[lr]h_thumb_cmc_roll",
        "[lr]h_thumb_cmc_pitch",
        "[lr]h_index_mcp_pitch",
        "[lr]h_middle_mcp_pitch",
        "[lr]h_ring_mcp_pitch",
        "[lr]h_pinky_mcp_pitch",
    ],
    effort_limit_sim=30.0,
    velocity_limit_sim=10.0,
    stiffness=10.0,
    damping=0.2,
    armature=0.001,
)

G1_L6_CFG.actuators["l6_couplings"] = ImplicitActuatorCfg(
    joint_names_expr=["[lr]h_.*_dip"],
    effort_limit_sim=30.0,
    velocity_limit_sim=10.0,
    stiffness=0.0,
    damping=0.01,
    armature=0.001,
)
