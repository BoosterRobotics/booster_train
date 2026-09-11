"""Replay retargeted Booster robot motions from NPZ files.

.. code-block:: bash

    # Usage - Direct file path
    python replay_npz.py --robot t2 --motion <path_to_motion.npz>
    
    # Usage - From wandb registry
    python replay_npz.py --robot k1 --registry_name <wandb_registry_name>
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import pathlib
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--motion", type=str, default=None, help="Path to the motion npz file.")
parser.add_argument("--registry_name", type=str, default=None, help="The name of the wand registry.")
parser.add_argument(
    "--robot",
    type=str,
    choices=("k1", "t1", "t2"),
    default="k1",
    help="The Booster robot model targeted by the motion data.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from booster_train.tasks.manager_based.beyond_mimic.mdp.commands import MotionLoader

if args_cli.robot == "k1":
    from booster_train.assets.robots.booster_k1 import BOOSTER_K1_CFG as ROBOT_CFG
elif args_cli.robot == "t1":
    from booster_train.assets.robots.booster_t1 import BOOSTER_T1_CFG as ROBOT_CFG
elif args_cli.robot == "t2":
    from booster_train.assets.robots.booster_t2 import BOOSTER_T2_CFG as ROBOT_CFG
else:
    raise ValueError(f"Unknown robot: {args_cli.robot}")


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def resolve_motion_file() -> str:
    """Resolve a local motion path from the CLI arguments."""
    if args_cli.motion:
        motion_file = args_cli.motion
        if not os.path.isfile(motion_file):
            raise FileNotFoundError(f"Motion file not found: {motion_file}")
        return motion_file

    if args_cli.registry_name:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is required when using --registry_name. Install it with: pip install wandb"
            ) from exc

        registry_name = args_cli.registry_name
        if ":" not in registry_name:
            registry_name += ":latest"
        artifact = wandb.Api().artifact(registry_name)
        return str(pathlib.Path(artifact.download()) / "motion.npz")

    raise ValueError("Either --motion or --registry_name must be provided.")


def load_motion_fps(motion_file: str) -> float:
    """Read and validate the replay frequency stored in an NPZ file."""
    with np.load(motion_file) as data:
        if "fps" not in data:
            raise KeyError(f"Motion file does not contain 'fps': {motion_file}")
        fps_values = np.asarray(data["fps"]).reshape(-1)
        if fps_values.size != 1:
            raise ValueError(f"Motion file 'fps' must contain one value, got shape {data['fps'].shape}")
        fps = float(fps_values[0])
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"Motion file has invalid fps: {fps}")
    return fps


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, motion_file: str):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    robot_joint_indexes = robot.find_joints(robot.joint_names, preserve_order=True)[0]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    root_body_name = "trunk"
    if root_body_name not in robot.body_names:
        raise ValueError(f"{args_cli.robot.upper()} asset does not contain root body '{root_body_name}'.")
    try:
        motion = MotionLoader(
            motion_file,
            [root_body_name],
            robot.joint_names,
            default_motion_body_names=robot.body_names,
            default_motion_joint_names=robot.joint_names,
            tail_len=0,
            device=str(sim.device),
        )
    except (AssertionError, IndexError, KeyError, ValueError) as exc:
        raise ValueError(
            f"Motion '{motion_file}' is not compatible with {args_cli.robot.upper()}. "
            "Retarget the motion to the selected robot and ensure body_names and joint_names are present."
        ) from exc
    if motion.time_step_total == 0:
        raise ValueError(f"Motion file contains no frames: {motion_file}")
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    # Simulation loop
    while simulation_app.is_running():
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion.joint_pos[time_steps]
        joint_vel[:, robot_joint_indexes] = motion.joint_vel[time_steps]
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        time_steps += 1
        time_steps[time_steps >= motion.time_step_total] = 0


def main():
    motion_file = resolve_motion_file()
    motion_fps = load_motion_fps(motion_file)
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / motion_fps
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene, motion_file)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
