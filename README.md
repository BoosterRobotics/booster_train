# Booster train

## Overview

This repository provides a set of reinforcement learning tasks for Booster robots using [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html).
Currently it includes the fabulous [BeyondMimic motion tracking](https://github.com/HybridRobotics/whole_body_tracking) framework, adapted for Booster robots, with support for the T2, T1, and K1 platforms.
This repository follows the standard Isaac Lab project structure, and is tested with IsaacLab 2.3.2  and Isaac Sim 5.1.0

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):
    ```bash
    git clone https://github.com/BoosterRobotics/booster_train.git
    ```

- Download and install booster_assets:
   - Clone the [booster_assets](https://github.com/BoosterRobotics/booster_assets) which contains Booster robot models and motion data.
   - Install booster_assets python helper following the instructions in the repository.

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/booster_train
    ```

- Prepare BeyondMimic motion data:
    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/csv_to_npz.py --headless --input_file=<PATH_TO_BOOSTER_ASSETS>/motions/K1/<MOTION>.csv --input_fps=<FPS> --output_file=<PATH_TO_BOOSTER_ASSETS>/motions/K1/<MOTION>.npz
    ```

## NOTES on T2 Sim2real
### 1. Armature and T-N curve constraints

`armature` is the actuator's equivalent inertia reflected to the joint, affecting joint acceleration and PD response. 

The T-N (torque-speed) curve constrains the torque available at a given speed. A constant torque limit alone allows unrealistically large torques at high speeds, which a policy can exploit during fast swings, recovery, and impacts. [`BoosterDelayedPDActuator`](source/booster_train/booster_train/assets/robots/actuator.py) implements a symmetric, piecewise-linear approximation:

- For `abs(joint_velocity) <= knee_point_velocity`, the torque magnitude is limited to `effort_limit`.
- Between `knee_point_velocity` and `velocity_limit`, the available torque decreases linearly to zero.
- At or above `velocity_limit`, the available torque is zero in this model, including braking torque; this is a simplified envelope rather than a full electrical motor model.

Unless explicitly overridden, the joint configuration derives PD gains from `Kp = J_eq * (2 * pi * f_n)**2` and `Kd = 2 * zeta * J_eq * (2 * pi * f_n)`, using `armature` as `J_eq`. The current T2 defaults are `f_n=8 Hz` and `zeta=2.0`.

### 2. Randomize joint command delay

We use delay randomization of up to 8 decimation steps. The policy runs at **50 Hz**, and each action uses `decimation=4`, corresponding to 4 physics simulation steps. Command delay introduces feedback lag; randomizing it improves robustness to communication and actuation latency on hardware. At reset, delays of `0–8` physics steps are sampled per environment and actuator group. With a `5 ms` physics timestep, this corresponds to **0–40 ms**.

Recheck the delay range whenever the physics timestep changes. Keep the deployment policy frequency consistent with training, currently **50 Hz**.

### 3. Handle parallel joints consistently in simulation and deployment

Use `BoosterT2WaistParaWrapperCfg` for waist pitch/roll and `BoosterT2AnkleParaWrapperCfg` for ankle pitch/roll, with **`serial_index=0` for pitch and `serial_index=1` for roll**. These wrappers convert the one-motor `base_joint_cfg` into the equivalent joint-side limits and inertia used by simulation.

The core of this conversion is a local linearization of the parallel-joint kinematics: the mechanism Jacobian is simplified around the neutral pose and reduced to fixed conversion ratios. With `J = ∂q_m/∂q`, the local model is `q_m - q_m0 ~= J(q - q0)`. Because the Jacobian changes with posture, the neutral-pose approximation is stored as fixed effective/calibrated coefficients for runtime use.

`ParallelJointWrapperCfg.__post_init__` selects the coefficient with `serial_index` and applies four direct conversions:

```text
effort_limit          = effort_ratio[idx]               * base_joint_cfg.effort_limit
velocity_limit        = velocity_ratio[idx]             * base_joint_cfg.velocity_limit
knee_point_velocity   = knee_point_velocity_ratio[idx]  * base_joint_cfg.knee_point_velocity
armature              = armature_ratio[idx]             * base_joint_cfg.armature
```

Under this local model, virtual work gives `tau_joint = J^T * tau_motor`, velocity gives `q_m_dot = J * q_dot`, and reflected inertia gives `I_joint = J^T * I_motor * J` (or `sum(Ii*Ji^2)` for one scalar joint). Thus `effort_ratio` represents effective torque transmission, `velocity_ratio` and `knee_point_velocity_ratio` represent effective speed transmission, and `armature_ratio` represents reflected inertia. The implementation stores the neutral-pose approximation as fixed ratios and applies them directly to the base configuration.

After these fields are transformed, `super().__post_init__()` computes default simulation gains from the transformed inertia: `Kp = armature * (2*pi*f_n)^2` and `Kd = 2*zeta*armature*(2*pi*f_n)`. The wrapper only changes configuration values; it does not perform a runtime coordinate or torque transformation. The optional simulation limit multipliers are applied separately by `BoosterDelayedActuatorCfg`.

The conversion is implemented in [`ParallelJointWrapperCfg.__post_init__`](source/booster_train/booster_train/assets/robots/actuator.py#L266), with T2 coefficients defined in the [waist](source/booster_train/booster_train/assets/robots/actuator.py#L477) and [ankle](source/booster_train/booster_train/assets/robots/actuator.py#L489) wrappers.

> **DEPLOYMENT:** For these four axes, **never copy the simulation `Kd` to the robot**: the [default simulation damping](source/booster_train/booster_train/assets/robots/actuator.py#L165) includes `armature_ratio`, while deployment `robot.joint_damping` is sent directly to the motor. Recompute **`Kd = 2 * zeta * J_eq_motor * (2 * pi * f_n)`** using motor-side inertia, **without `armature_ratio`**.

### 4. Keep PD gains and action scaling consistent

`T2_ACTION_SCALE = 0.25 * effort_limit / Kp` converts policy outputs into position offsets. Recheck PD gains and action scaling after changing actuator parameters; later configuration overrides do not automatically update the scale. Use the training-time action scale during deployment.

## Usage

- Listing the available tasks:

    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/list_envs.py
    ```

- Running a task:

    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/rsl_rl/train.py --task=<TASK_NAME> --headless --device cuda:N
    ```

- Play a trained policy and export it for deployment:

    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/rsl_rl/play.py --task=<TASK_NAME> --checkpoint=<CHECKPOINT_PATH>
    ```

    This script also exports the trained policy to a TorchScript/ONNX file for deployment on real robots in `logs/rsl_rl/<EXPERIMENT>/<RUN>/exported/`.

## Deploy

After a model has been trained and exported, you can deploy the trained policy in MuJoCo or on real Booster robots using the [booster_deploy](https://github.com/BoosterRobotics/booster_deploy) repository. For more details, please refer to the instructions in the [booster_deploy](https://github.com/BoosterRobotics/booster_deploy) repository.


## Acknowledgements

- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking): the motion tracking training in BeyondMimic, which is a versatile humanoid control framework that provides highly dynamic motion tracking.
