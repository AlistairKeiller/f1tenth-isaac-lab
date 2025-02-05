from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform
from isaaclab.actuators import ActuatorNetLSTMCfg, DCMotorCfg


class F1tenthEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 20.0
    action_scale = 0.5
    action_space = 12  # TODO
    observation_space = 48  # TODO
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene

    # events

    # robot
    robot: ArticulationCfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"F1tenth.usd",  # TODO
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
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
            init_state=ArticulationCfg.InitialStateCfg( # TODO
                pos=(0.0, 0.0, 0.6),
                joint_pos={
                    ".*HAA": 0.0,  # all HAA
                    ".*F_HFE": 0.4,  # both front HFE
                    ".*H_HFE": -0.4,  # both hind HFE
                    ".*F_KFE": -0.8,  # both front KFE
                    ".*H_KFE": 0.8,  # both hind KFE
                },
            ),
            actuators={ # TODO
                "legs": ActuatorNetLSTMCfg(
                    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
                    network_file=f"/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt",
                    saturation_effort=120.0,
                    effort_limit=80.0,
                    velocity_limit=7.5,
                )
            },
        )
    )


# class F1tenthEnv(DirectRLEnv):
