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
