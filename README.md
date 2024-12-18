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

Input topics : 
- `[namespace]/rgb` : `Ìmage` message (`bgr8` format)
- `[namespace]/depth` : `Image` message (`mono16` format)
- `[namespace]/pcl` : `PointCloud2` message (`.data` buffer is expected to contain 3 or 6 fields of `float32` such that it can be reshaped to a numpy array of size `[image_height, image_width, n_fields]` that offers pixel correspondance with the `rgb` and `depth` image)

Output topics :
- `[namespace]/human` : `TransformStamped` message with detected human position

Output tf :
- `TransformStamped` with child frame id `[namespace]/human`

## Options
Flags to run with options

See `python rgbd_detect_3d_dir.py -h`

Use `export VERBOSE=0` to set minimal verbosity (warnings, debug, errors and success). Level 1 adds info and 2 adds timing info.

## Acknowledgments
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
- Add links and citations in Acknowledgments section