# Booster RL Tasks

## Overview

This repository provides a set of reinforcement learning tasks for Booster robots using [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html).
Currently it includes the fabulous [BeyondMimic motion tracking](https://github.com/HybridRobotics/whole_body_tracking) framework adapted to Booster K1 robots.
The motion conversion and replay utilities under `scripts/mimic/` support both Booster K1 and T1 robots.
This repository follows the standard Isaac Lab project structure, and is tested with IsaacLab 2.2 and Isaac Sim 5.0.

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
    python scripts/mimic/csv_to_npz.py --headless --input_file=<PATH_TO_BOOSTER_ASSETS>/motions/<ROBOT>/<MOTION>.csv --input_fps=<FPS> --output_file=<PATH_TO_BOOSTER_ASSETS>/motions/<ROBOT>/<MOTION>.npz --output_fps=50 --robot=<k1|t1>
    ```

    Optional arguments:

    - `--robot=t1` converts T1 motion files. The default robot is `k1`.
    - `--frame_range <START> <END>` converts only a subset of frames. Frame indices are 1-based and inclusive.
    - `--output_fps` controls the interpolation rate of the exported `.npz` motion.

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

- Replay a converted motion file for inspection:

    ```bash
    python scripts/mimic/replay_npz.py --motion=<PATH_TO_MOTION>.npz --robot=<k1|t1>

    # or download and replay from Weights & Biases registry
    python scripts/mimic/replay_npz.py --registry_name=<WANDB_REGISTRY_NAME> --robot=<k1|t1>
    ```

    When `--registry_name` does not include an alias, the script automatically uses `:latest`.

## Deploy

After a model has been trained and exported, you can deploy the trained policy in MuJoCo or on real Booster robots using the [booster_deploy](https://github.com/BoosterRobotics/booster_deploy) repository. For more details, please refer to the instructions in the [booster_deploy](https://github.com/BoosterRobotics/booster_deploy) repository.


## Acknowledgements

- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking): the motion tracking training in BeyondMimic, which is a versatile humanoid control framework that provides highly dynamic motion tracking.