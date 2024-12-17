ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
COPY ./cuda-keyring_1.0-1_all.deb cuda-keyring_1.0-1_all.deb
RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list && dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-get update

RUN apt-get install -y software-properties-common
RUN apt-get update
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt install -y gcc-9
RUN apt-get install libstdc++6
RUN apt-get update


ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# Refer a advise in the issues page
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install xtcocotools
RUN pip install cython
RUN pip install xtcocotools

# Install MMCV
RUN pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html


#System full upgrade
RUN apt-get update && apt-get --with-new-pkgs upgrade -y


#Install for python rgbdetect
RUN pip install mmdet==2.28.2
RUN git clone https://github.com/ViTAE-Transformer/ViTPose/
WORKDIR /workspace/ViTPose
RUN pip install -v -e .
RUN pip install timm==0.4.9 einops
RUN pip install print-color
RUN pip install --extra-index-url https://rospypi.github.io/simple/ rospy
RUN pip install -U --extra-index-url https://rospypi.github.io/simple/_pre sensor_msgs tf2_ros tf2_sensor_msgs tf tf2-py
RUN pip install -U --extra-index-url https://rospypi.github.io/simple/_pre cv_bridge
RUN pip install albumentations==1.1.0
RUN pip install onnxruntime
RUN pip uninstall opencv-python -y
RUN pip install opencv-python==4.10.0.84

WORKDIR /workspace/rgbd_pose_and_depth
