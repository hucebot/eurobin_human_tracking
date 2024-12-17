#!/bin/bash
IsRunning=`docker ps -f name=rgbd_detect | grep -c "rgbd_detect"`;
if [ $IsRunning -eq "0" ]; then
    xhost +local:docker
    docker run --rm \
        --gpus all \
        -e DISPLAY=$DISPLAY \
        -e XAUTHORITY=$XAUTHORITY \
        -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        -e 'QT_X11_NO_MITSHM=1' \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v /tmp/docker_share:/tmp/docker_share \
        -v `pwd`:/workspace/rgbd_pose_and_depth \
        --ipc host \
        --device /dev/dri \
        --device /dev/snd \
        --device /dev/input \
        --device /dev/bus/usb \
        --privileged \
        --ulimit rtprio=99 \
        --net host \
        --name rgbd_detect \
        --entrypoint /bin/bash \
        -ti inria_docker:rgbd_detect
else
    echo "Docker image is already running. Opening new terminal...";
    docker exec -ti rgbd_detect /bin/bash
fi