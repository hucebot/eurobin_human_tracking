# euROBIN Human Tracking

Human tracking pipeline used in euROBIN system as described in XXX.

## Description

The pipeline consists of the following main components :
- Human detection
- Human tracking
- Pose estimation (2D)
- Gaze estimation

## Installation
How to install with docker and launch interactively

- Build docker

```bash
docker build -t inria_docker:rgbd_detect .
```

- Launch docker

```bash
sh launch.sh
```

## ROS Interface
Input / output topics and format

## Options
Flags to run with options

## Acknoledgments
MMDet, MMPose, 6DRepNet...


## TODO
- Clean the code of unnecesary options (keep only sixdrep)
- Remove unecessary dependencies
- Keep only rgbd_detect_3d_dir.py
- Add models in external links, LFS or directly from their source in the corresponding repos
- Add a description of all options
- Add ROS interface description
- Add illustration
- Add instructions and `requirements.txt` for local installation