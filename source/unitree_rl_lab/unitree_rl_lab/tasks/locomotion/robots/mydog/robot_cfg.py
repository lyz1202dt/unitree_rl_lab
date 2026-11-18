"""Articulation configuration for the 03dog robot used by the mydog task."""

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass


DOG03_URDF_PATH = os.environ.get(
    "DOG03_URDF_PATH",
    str(Path(__file__).resolve().parent / "assets" / "03dog" / "urdf" / "03dog.urdf"),
)


@configclass
class Dog03ArticulationCfg(ArticulationCfg):
    """Articulation with an explicit joint order for deployment export."""

    joint_sdk_names: list[str] = None


DOG03_CFG = Dog03ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=False,
        asset_path=DOG03_URDF_PATH,
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
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.40),
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": -0.7854,
            ".*_calf_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=30.0,
            velocity_limit=30.0,
            stiffness=40.0,
            damping=1.0,
            friction=0.32,
            dynamic_friction=0.29274320,
            viscous_friction=0.02658733,
            min_delay=1,
            max_delay=2,
        ),
    },
    joint_sdk_names=[
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ],
)
