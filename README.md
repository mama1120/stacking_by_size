# Robot programming HKA WS 2024

This repository contains the code for the robot programming course (the policy learning part) at the University of Applied Sciences Karlsruhe.

We will use and update this repository throughout the course. Hello.

## Quick start

### Environment setup

**Requirements:** have docker installed including the post-installation steps.

**Note:** The default settings are for nvidia GPU support. If you don't have an nvidia GPU, open up `build_image.sh` and set the `render` argument to `base`. Also, remove the `--gpus all` flag from the `docker run` command in `run_container.sh`.

Build the docker image with

```bash
./build_image.sh
```

Run the container with
```bash
./run_container.sh
```

Check whether you can open a window from the container by running
```bash
python3 stack_cubes.py
```
To test the tensorflow functionality:
Check whether you can open a window from the container by running
```bash
python3 test_model.py
```

To create a new dataset with random positions, run:
```bash
python3 auto_create_dataset.py
```

To create a new dataset with random positions and orientations, run:
```bash
python3 auto_create_dataset_orientations.py
```

To create a new dataset with random positions with noise, run:
```bash
python3 auto_noise_create_dataset.py
```

Combine the future state in the last state file:
```bash
python3 combine_json.py
```

Train the model with only regression:
```bash
python3 train_model.py
```

Train the model with regression for position and orientation and classification for gripper:
```bash
python3 train_model_binary.py
```

Train the model with regression for position and orientation and classification for gripper. Modified for testing different networks:
```bash
python3 train_model_binary_mod.py
```

Test the model in the enviromment:
```bash
python3 env_test_model.py
```
To train with dagger:
```bash
python3 model_largest_cube_DAgger.py
```
DAgger_env.py has many functions to use the different scripts. They include the functions to use the model, the expert definition and some enviroment.