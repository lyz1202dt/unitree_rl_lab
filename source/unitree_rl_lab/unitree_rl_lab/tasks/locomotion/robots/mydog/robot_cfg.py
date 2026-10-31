"""Articulation configuration for the DIAN robot used by the mydog task."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass


DIAN_USD_PATH = os.environ.get(
    "DIAN_USD_PATH", "/home/hi/下载/点足总装修订版10-18/DIAN/usd/DIAN.usd"
)


@configclass
class DianArticulationCfg(ArticulationCfg):
    """Articulation with an explicit joint order for deployment export."""

    joint_sdk_names: list[str] = None


DIAN_CFG = DianArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=DIAN_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.44),
        joint_pos={
            "FL_J1": 0.0,
            "FL_J2": 0.9,
            "FL_J3": -0.5,
            "FR_J1": 0.0,
            "FR_J2": -0.9,
            "FR_J3": 0.5,
            "RL_J1": 0.0,
            "RL_J2": 0.7,
            "RL_J3": -0.5,
            "RR_J1": 0.0,
            "RR_J2": -0.7,
            "RR_J3": 0.5,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*_J[123]"],
            effort_limit=25.0,
            velocity_limit=30.0,
            stiffness=40.0,
            damping=0.4,
            min_delay=0,
            max_delay=0,  # Restore 2 for random 0/5/10 ms delay.
        ),
    },
    joint_sdk_names=[
        "FL_J1", "FL_J2", "FL_J3",
        "FR_J1", "FR_J2", "FR_J3",
        "RL_J1", "RL_J2", "RL_J3",
        "RR_J1", "RR_J2", "RR_J3",
    ],
)
