# euROBIN Human Tracking

Human tracking pipeline used in euROBIN system as described in XXX.

![Pipeline visu](images/eurobin_pipeline_gif.gif)

## Description

The pipeline consists of the following main components :
- Human detection
- Human tracking
- Pose estimation (2D)
- Gaze estimation

## Installation
How to install with docker and launch interactively

- Get the appropriate cuda keyring file (`cuda-keyring_1.0-1_all.deb`) that may be required to build the docker from [nvidia](https://developer.download.nvidia.com/compute/cuda/repos), optionally you can try without it by commenting the corresponding lines in `Dockerfile`

- Download models and place them in `./models/`
```
wget https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-421362b6.pth
mv yolov3_d53_320_273e_coco-421362b6.pth models/
```

Go to : [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccifT1XlGRatxg3vw?e=9wz7BY) and download the model then put in under `models/vitpose_small.pth`

The 6DRepNet models (face detection and 6D pose estimation) are already in `/models` (downloaded from [SixDRepNet/weights](https://github.com/jahongir7174/SixDRepNet/tree/master/weights))

- Build docker

```bash
docker build -t inria_docker:rgbd_detect .
```

- Launch docker

```bash
sh launch.sh
```

- Launch the pipeline

```bash
python rgbd_detect_3d_dir.py -sixdrep
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
The implementation is based on the following implementations of human detection and tracking, human pose estimation and 6D Head Pose estimation

- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [MMPose](https://github.com/open-mmlab/mmpose)
- [6DRepNet](https://github.com/thohemp/6DRepNet) and [SixDRepNet](https://github.com/jahongir7174/SixDRepNet)

## TODO
- Add a description of all options
- Add instructions and `requirements.txt` for local installation