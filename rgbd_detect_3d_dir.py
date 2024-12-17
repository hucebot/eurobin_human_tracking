#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO : sort imports

# mmdet and mmpose import
from mmpose.apis import (
    get_track_id,
    inference_top_down_pose_model,
    init_pose_model,
    process_mmdet_results,
)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

# ros related import
import rospy
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import tf2_ros

# other import
import cv2
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from datetime import datetime
import time
import json
import warnings
import numpy as np
from PyKDL import Rotation
import copy
# import imageio
from PIL import Image as PILImage

from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    has_mb = True
    # motion bert import
    from lib.utils.tools import *
    from lib.utils.learning import *
    from lib.utils.utils_data import flip_data
    from lib.data.dataset_wild import WildDetDataset
    from lib.utils.vismo import render_and_save
except:
    has_mb = False
    prWarning("No MotionBERT import, fail")
    
try:
    # gafa import (optional)
    has_gafa = True
    from gafa_utils import body_transform, head_transform, head_transform_rest, normalize_img, body_transform_from_bb, normalize_img_torch, head_transform_face
    from gafa.models.gazenet import GazeNet
except:
    has_gafa = False
    prWarning("No GAFA import, fail")
    from gafa_utils import body_transform, head_transform, head_transform_rest, normalize_img, body_transform_from_bb, normalize_img_torch, head_transform_face
    
# 6D Rep import
try:
    has_sixdrep = True
    from sixdrep.util import FaceDetector, compute_euler
    from sixdrep.utils import sixdreptransform
except:
    has_sixdrep = False
    prWarning("No 6D Rep import, fail")
    
    
# gaze estimation simple models import
try:
    has_gaze_est = True
    from gaze_estimation.models import resnet18, mobilenet_v2, mobileone_s0
    from gaze_estimation.utils import pre_process
    from gaze_estimation.models import SCRFD
except:
    has_gaze_est = False
    prWarning("No GazeEst import, fail")


# remove numpy scientific notation
np.set_printoptions(suppress=True)


class InferenceNodeRGBD(object):
    def __init__(self, args):

        # init args
        self.args = args

        # init detector and pose
        self.det_model = init_detector(
            args.det_config, args.det_checkpoint, device=args.device.lower()
        )

        self.pose_model = init_pose_model(
            args.pose_config, args.pose_checkpoint, device=args.device.lower()
        )

        # if enabled, init MotionBERT
        if self.args.use_mb:
            # init 3d MotionBERT model
            prInfo('Initialiazing 3D Pose Lifter {}'.format(args.mb_checkpoint))        
            mb_3d_args = get_config(args.mb_3d_config)
            self.motionbert_3d_model = load_backbone(mb_3d_args)
            if torch.cuda.is_available():
                self.motionbert_3d_model = nn.DataParallel(self.motionbert_3d_model)
                self.motionbert_3d_model = self.motionbert_3d_model.cuda()
            else:
                prError("Expect cuda to be available but is_available returned false")
                exit(0)

            prInfo('Loading checkpoint {}'.format(args.mb_checkpoint))
            mb_checkpoint = torch.load(args.mb_checkpoint, map_location=lambda storage, loc: storage)
            self.motionbert_3d_model.load_state_dict(mb_checkpoint['model_pos'], strict=True)
            self.motionbert_3d_model.eval()
            prInfo('Loaded motionbert_3d_model')
            # no need for the whole WildDetDataset stuff, just manually make the input trajectories for the tracks

        # if enabled, init GAFA
        if self.args.use_gafa:
            self.gafa_model = GazeNet(n_frames=self.args.gafa_n_frames)
            self.gafa_model.load_state_dict(torch.load(
                self.args.gafa_checkpoint)) #, map_location=torch.device("cpu"))['state_dict'])

            self.gafa_model.cuda()
            self.gafa_model.eval()

            prInfo(
                "Loaded GAFA model from {}".format(
                    self.args.gafa_checkpoint))
        
        # if enabled, init gaze resnet
        if self.args.use_gaze_resnet:
            self.face_detector = SCRFD(model_path="./gaze_estimation/weights/det_10g.onnx")
            self.gaze_estimation_model = resnet18(pretrained = False, num_classes = 90)     
            state_dict = torch.load("./gaze_estimation/weights/resnet18.pt", map_location=args.device.lower())
            self.gaze_estimation_model.load_state_dict(state_dict)
            self.gaze_estimation_model.to(args.device.lower())
            self.gaze_estimation_model.eval()
            prInfo('Loaded ResNet18 for gaze estimation')
        
        # if enabled, init 6DRep
        if self.args.use_six_d_rep:
            
            self.sixdrep_model = torch.load(f='./sixdrep/weights/best.pt', map_location='cuda')
            self.sixdrep_model = self.sixdrep_model['model'].float().fuse()
            self.sixdrep_detector = FaceDetector('./sixdrep/weights/detection.onnx')
    
            self.sixdrep_model.half()
            self.sixdrep_model.eval()

        
        # dataset for detection and pose
        self.dataset = self.pose_model.cfg.data["test"]["type"]
        self.dataset_info = self.pose_model.cfg.data["test"].get(
            "self.dataset_info", None
        )
        if self.dataset_info is None:
            warnings.warn(
                "Please set `self.dataset_info` in the config."
                "Check https://github.com/open-mmlab/mmpose/pull/663 for details.",
                DeprecationWarning,
            )
        else:
            self.dataset_info = DatasetInfo(self.dataset_info)

        self.return_heatmap = False

        # variables to keep tracks along time or in the current frame
        self.next_id = 0
        self.pose_results = []
        self.tracks_in_current_image = {}
        self.tracks = {} # all the tracks along time, we need to keep and history with some data
        
        # shared variables for the received images and pcl
        self.rgb = None  # Image frame
        self.depth = None  # Image frame

        self.pcl_array_rgb = None
        self.pcl_array_xyz = None

        # viewing options
        self.depth_array_max_threshold = 20000
        self.depth_cmap = get_mpl_colormap(args.depth_cmap)
        self.confidence_cmap = get_mpl_colormap("viridis")
        self.vis_img = None  # output image RGB + detections
        self.view_all_classes_dets = True
        self.display_all_detection = args.display_all_detection
        self.light_display = args.light_display
        
        # counter for the incoming frames
        self.pcl_current_seq = -1
        self.rgb_current_seq = -1
        self.last_inferred_seq = -1
        self.depth_current_seq = -1
        self.current_image_count = 0
        self.rgb_frame_id = None # received from ROS image

        # CV Bridge for receiving frames
        self.br = CvBridge()

        # Set ROS node rate
        prInfo("Setting node rate to {} fps".format(args.fps))
        self.loop_rate = rospy.Rate(args.fps)

        # create the output path
        now = datetime.now()
        timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
        self.save_dir = os.path.join("output", "record_{:s}".format(timestamp))
        self.metadata = os.path.join(self.save_dir, "metadata.json")
        self.save_dir_rgb = os.path.join(self.save_dir, "rgb")
        self.save_dir_depth = os.path.join(self.save_dir, "depth")
        self.save_dir_result = os.path.join(self.save_dir, "output")
        self.save_dir_pcl_bin = os.path.join(self.save_dir, "pcl")

        if args.save or args.light_save:
            prInfo(
                "Saving to {}/[rgb][depth][depth_color][output][pcl]".format(
                    self.save_dir
                )
            )
            if not os.path.exists(self.save_dir):
                prInfo(
                    "Creating directories to {}/[rgb][depth][depth_color][output][pcl]".format(
                        self.save_dir
                    )
                )
                os.makedirs(self.save_dir)
                os.makedirs(self.save_dir_rgb)
                os.makedirs(self.save_dir_pcl_bin)

                if args.save:
                    os.makedirs(self.save_dir_depth)
                    os.makedirs(self.save_dir_result)

                args_dic = vars(args)
                with open(self.metadata, "w") as fp:
                    json.dump(args_dic, fp)

                prSuccess(
                    "Created directories to {}/[rgb][depth][depth_color][output][pcl]".format(
                        self.save_dir
                    )
                )
                time.sleep(1)

        # ROS publishers
        self.goal_pub = rospy.Publisher(
            args.namespace + "/human", TransformStamped, queue_size=1
        )

        self.tf_br = tf2_ros.TransformBroadcaster()

        # ROS subscribers
        rgb_topic = args.namespace + "/rgb"
        depth_topic = args.namespace + "/depth"
        pcl_topic = args.namespace + "/pcl"
        prInfo("Subscribing to {} for RGB".format(rgb_topic))
        rospy.Subscriber(rgb_topic, Image, self.callback_rgb)
        prInfo("Subscribing to {} for depth".format(depth_topic))
        rospy.Subscriber(depth_topic, Image, self.callback_depth)
        prInfo("Subscribing to {} for PCL".format(pcl_topic))
        rospy.Subscriber(pcl_topic, PointCloud2, self.callback_pcl)


    def callback_pcl(self, msg):
        if self.args.flip:
            pcl_array = np.frombuffer(msg.data, dtype=np.float32).reshape(
                (msg.height, msg.width, -1)
            )[::-1, ::-1, :]
        else:
            pcl_array = np.frombuffer(msg.data, dtype=np.float32).reshape(
                (msg.height, msg.width, -1)
            )

        # pcl_array = pcl_array[::-1, :, :]
        self.pcl_array_xyz = pcl_array[:, :, :3]
        # self.pcl_array_rgb = pcl_array[:,:,3:]
        self.pcl_current_seq = msg.header.seq
        # rospy.loginfo('pcl received ({})...'.format(msg.header.seq))

    def callback_rgb(self, msg):
        if self.rgb_frame_id != msg.header.frame_id:
            self.rgb_frame_id = msg.header.frame_id
        if self.args.flip:
            self.rgb = cv2.flip(self.br.imgmsg_to_cv2(msg, "bgr8"), -1)
        else:
            self.rgb = self.br.imgmsg_to_cv2(msg, "bgr8")

        # self.rgb = cv2.rotate(self.rgb, cv2.ROTATE_180)
        self.rgb_current_seq = msg.header.seq
        # rospy.loginfo('RGB received ({})...'.format(msg.header.seq))
        self.rgb_timestamp = msg.header.stamp

    def callback_depth(self, msg):
        if self.args.flip:
            self.depth = cv2.flip(self.br.imgmsg_to_cv2(msg, "mono16"), -1)
        else:
            self.depth = self.br.imgmsg_to_cv2(msg, "mono16")

        self.depth_current_seq = msg.header.seq
        # rospy.loginfo('Depth received ({})...'.format(msg.header.seq))

    def is_ready(self):
        ready = (
            (self.rgb is not None)
            and (self.depth is not None)
            and (self.pcl_array_xyz is not None)
        )
        return ready
    
    @timeit
    def save_rgb(self, image_count, image_seq_unique, timestamp):
        prWarning("Saving images here may suffer synchronization issues, use visualizer.py for lighter save")
        rgb_path = os.path.join(
            self.save_dir_rgb,
            "{:08d}_seq_{:010d}_ts_{}.png".format(
                image_count, image_seq_unique, timestamp
            ),
        )
        cv2.imwrite(rgb_path, self.rgb)
        prSuccess("Saved RGB to {}".format(rgb_path))

    @timeit
    def save_depth(self, image_count, image_seq_unique, timestamp):
        prWarning("Saving images here may suffer synchronization issues, use visualizer.py for lighter save")
        depth_path = os.path.join(
            self.save_dir_depth,
            "{:08d}_seq_{:010d}_ts_{}.png".format(
                image_count, image_seq_unique, timestamp
            ),
        )
        cv2.imwrite(depth_path, self.depth)
        prSuccess("Saved depth to {}".format(depth_path))

    @timeit
    def save_output_image(self, image_count, image_seq_unique, timestamp):
        prWarning("Saving images here may suffer synchronization issues, use visualizer.py for lighter save")
        results_path = os.path.join(
            self.save_dir_result,
            "{:08d}_seq_{:010d}_ts_{}.png".format(
                image_count, image_seq_unique, timestamp
            ),
        )
        cv2.imwrite(results_path, self.vis_img)
        prSuccess("Saved result to {}".format(results_path))

    @timeit
    def save_pcl(self, image_count, image_seq_unique, timestamp):
        prWarning("Saving images here may suffer synchronization issues, use visualizer.py for lighter save")
        pcl_path = os.path.join(
            self.save_dir_pcl_bin,
            "{:08d}_seq_{:010d}_ts_{}.bin".format(
                image_count, image_seq_unique, timestamp
            ),
        )
        self.pcl_array_xyz.tofile(pcl_path)
        prSuccess("Saved pcl to {}".format(pcl_path))

    @timeit
    def plot_mmdet_bbox(self, mmdet_results, array_shape):
        for c in range(len(mmdet_results)):
            if len(mmdet_results[c]) > 0:
                for bi in range(mmdet_results[c].shape[0]):
                    if mmdet_results[c][bi, 4] > self.args.bbox_thr:
                        bbox = (
                            mmdet_results[c][bi, :4]
                            .copy()
                            .astype(np.int32)
                        )
                        bbox_ints = [
                            int(bbox[0]),
                            int(bbox[1]),
                            int(bbox[2]),
                            int(bbox[3]),
                        ]
                        pt1 = (
                            min(
                                max(0, bbox_ints[0]),
                                array_shape[1],
                            ),
                            min(
                                max(0, bbox_ints[1]),
                                array_shape[0],
                            ),
                        )
                        pt2 = (
                            min(
                                max(0, bbox_ints[2]),
                                array_shape[1],
                            ),
                            min(
                                max(0, bbox_ints[3]),
                                array_shape[0],
                            ),
                        )
                        cv2.rectangle(
                            self.vis_img, pt1, pt2, (255, 255, 255), 1
                        )
                        cv2.putText(
                            self.vis_img,
                            "{:s} ({:.0f}%)".format(
                                YOLO_COCO_80_CLASSES[c],
                                mmdet_results[c][bi, 4] * 100,
                            ),
                            pt1,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5 * TEXT_SCALE,
                            (255, 255, 255),
                            1,
                        )

    @timeit           
    def plot_xyxy_person_bbox(self, idx, bbox, array_shape, track, poses_torso = None):
        bbox_ints = [
            int(bbox[0]),
            int(bbox[1]),
            int(bbox[2]),
            int(bbox[3]),
        ]
        pt1 = (
            min(max(0, bbox_ints[0]), array_shape[1]),
            min(max(0, bbox_ints[1]), array_shape[0]),
        )
        pt2 = (
            min(max(0, bbox_ints[2]), array_shape[1]),
            min(max(0, bbox_ints[3]), array_shape[0]),
        )
        color = RANDOM_COLORS[idx]
        # color_tuple = (int(color[0]), int(color[1]), int(color[2]))
        color_tuple = (255,255,255)
        
        # yolo score
        score = bbox[4]
        
        # current gaze
        if len(track["gaze_yaw_rad"]) > 0:
            yaw_g = int(np.rad2deg(track["gaze_yaw_rad"][-1]))
            pitch_g = int(np.rad2deg(track["gaze_pitch_rad"][-1]))
            if yaw_g == 180 or pitch_g == 180:
                yaw_g = "Unk"
                pitch_g = "Unk"
        else:
            yaw_g = "Unk"
            pitch_g = "Unk"
            
        # curent depth
        if len(track["depth_face"]) > 0:
            depth_f = track["depth_face"][-1]
        else:
            depth_f = "Unk"
        
        # position
        if poses_torso is not None:
            pose_body = np.array(poses_torso).mean(axis=0)
            pose_body_n = pose_body.copy()
            if type(pose_body) == np.ndarray:
                pose_body_n[0] = pose_body[2]
                pose_body_n[1] = -pose_body[0]
                pose_body_n[2] = -pose_body[1]
            else:
                pose_body_n = ["Unk", "Unk", "Unk"]
        else:
            pose_body_n = ["Unk", "Unk", "Unk"]
            
        # attention
        heat_count = 0
        history = 20
        length = min(len(track["depth_face"]), history)
        for i in range(length):
            index = len(track["depth_face"]) - i - 1
            yaw = np.rad2deg(track["gaze_yaw_rad"][index])
            pitch = np.rad2deg(track["gaze_pitch_rad"][index])
            depth = track["depth_face"][index]
                        
            thresh = (int(depth / 1000) + 1) * 3 # 3 deg per meter
            if np.abs(yaw) < thresh and pitch < 0:
                heat_count += 1
                # suppose we are looking down
        
        attention_score = int( min((heat_count * 2 / history), 1) * 100)
        
        draw_bbox_with_corners(self.vis_img, bbox_ints, color = color_tuple, thickness = 5, proportion = 0.2)
        
        text = "Person : {}% | Attention : {}%".format(score, attention_score)
        if poses_torso is not None and  type(pose_body) == np.ndarray:
            text2 = "Yaw = {} | Pitch = {} | pos = ({:.2f}, {:.2f}, {:.2f})".format(yaw_g, pitch_g, pose_body_n[0], pose_body_n[1], pose_body_n[2])
        
        cv2.putText(
            self.vis_img,
            text,
            (bbox_ints[0], bbox_ints[1] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * TEXT_SCALE,
            color_tuple,
            1,
        )
        
        if poses_torso is not None  and  type(pose_body) == np.ndarray:
            cv2.putText(
                self.vis_img,
                text2,
                (bbox_ints[0], bbox_ints[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35 * TEXT_SCALE,
                color_tuple,
                1,
            )

        # cv2.rectangle(self.vis_img, pt1, pt2, color_tuple, 2)
             
    @timeit   
    def process_keypoints(self, keypoints, depth_array, idx):
        body_center_joints = (
            []
        )  # to store center of lsho, rsho, lhip, rhip in pixels
        color = RANDOM_COLORS[idx]
        # color_tuple = (int(color[0]), int(color[1]), int(color[2]))
        color_tuple = (255,255,255)

        for j in range(keypoints.shape[0]):

            kp = keypoints[j, :]
            confidence = int(kp[2] * 255)
            confidence_color = (
                self.confidence_cmap[min(255, confidence)] * 255
            ).astype(np.uint8)

            if (
                kp[2] > self.args.kpt_thr
                and kp[0] > 0
                and kp[1] > 0
                and kp[0] < depth_array.shape[1]
                and kp[1] < depth_array.shape[0]
            ):

                if (j == 5) or (j == 6) or (j == 11) or (j == 12):
                    # one keypoint of the torso
                    body_center_joints.append(kp)

                if not self.args.no_show and not self.args.light_display:
                    # kp_color_tuple = (int(confidence_color[0]), int(confidence_color[1]), int(confidence_color[2]))
                    cv2.circle(
                        self.vis_img,
                        (int(kp[0]), int(kp[1])),
                        2,
                        color_tuple,
                        thickness=3,
                    )

                # if wrists, find depth and pose

                if j == 10:
                    # right wrist
                    depth_wrist = depth_array[int(kp[1]), int(kp[0])]
                    pose_wrist = self.pcl_array_xyz[
                        int(kp[1]), int(kp[0]), :
                    ]
                    self.tracks_in_current_image[idx][
                        "right_wrist_depth"
                    ] = depth_wrist
                    self.tracks_in_current_image[idx][
                        "right_wrist_pose"
                    ] = pose_wrist
                    if not self.light_display and not self.args.no_show:
                        cv2.drawMarker(
                            self.vis_img,
                            (int(kp[0]), int(kp[1])),
                            color=color_tuple,
                            thickness=3,
                            markerType=cv2.MARKER_CROSS,
                            line_type=cv2.LINE_AA,
                            markerSize=8,
                        )
                        # cv2.putText(
                        #     self.vis_img,
                        #     "{:.0f}cm | {:.2f} {:.2f} {:.2f}".format(
                        #         depth_wrist / 10,
                        #         pose_wrist[0],
                        #         pose_wrist[1],
                        #         pose_wrist[2],
                        #     ),
                        #     (int(kp[0]), int(kp[1])),
                        #     cv2.FONT_HERSHEY_SIMPLEX,
                        #     0.5 * TEXT_SCALE,
                        #     (255,255,255),
                        #     2,
                        # )
                        # cv2.putText(
                        #     self.vis_img,
                        #     "{:.0f}cm | {:.2f} {:.2f} {:.2f}".format(
                        #         depth_wrist / 10,
                        #         pose_wrist[0],
                        #         pose_wrist[1],
                        #         pose_wrist[2],
                        #     ),
                        #     (int(kp[0]), int(kp[1])),
                        #     cv2.FONT_HERSHEY_SIMPLEX,
                        #     0.5 * TEXT_SCALE,
                        #     color_tuple,
                        #     1,
                        # )

                elif j == 9:
                    # left wrist
                    depth_wrist = depth_array[int(kp[1]), int(kp[0])]
                    pose_wrist = self.pcl_array_xyz[
                        int(kp[1]), int(kp[0]), :
                    ]
                    self.tracks_in_current_image[idx][
                        "left_wrist_depth"
                    ] = depth_wrist
                    self.tracks_in_current_image[idx][
                        "left_wrist_pose"
                    ] = pose_wrist
                    if not self.light_display and not self.args.no_show:
                        cv2.drawMarker(
                            self.vis_img,
                            (int(kp[0]), int(kp[1])),
                            color=color_tuple,
                            thickness=3,
                            markerType=cv2.MARKER_CROSS,
                            line_type=cv2.LINE_AA,
                            markerSize=8,
                        )
                        # cv2.putText(
                        #     self.vis_img,
                        #     "{:.0f}cm | {:.2f} {:.2f} {:.2f}".format(
                        #         depth_wrist / 10,
                        #         pose_wrist[0],
                        #         pose_wrist[1],
                        #         pose_wrist[2],
                        #     ),
                        #     (int(kp[0]), int(kp[1])),
                        #     cv2.FONT_HERSHEY_SIMPLEX,
                        #     0.5 * TEXT_SCALE,
                        #     (255,255,255),
                        #     2,
                        # )
                        # cv2.putText(
                        #     self.vis_img,
                        #     "{:.0f}cm | {:.2f} {:.2f} {:.2f}".format(
                        #         depth_wrist / 10,
                        #         pose_wrist[0],
                        #         pose_wrist[1],
                        #         pose_wrist[2],
                        #     ),
                        #     (int(kp[0]), int(kp[1])),
                        #     cv2.FONT_HERSHEY_SIMPLEX,
                        #     0.5 * TEXT_SCALE,
                        #     color_tuple,
                        #     1,
                        # )
                        
        return body_center_joints

    @timeit
    def get_depth_and_poses_of_torso(self, depth_array, lsho, rsho, lhip, rhip, idx):

        color = RANDOM_COLORS[idx]
        # color_tuple = (int(color[0]), int(color[1]), int(color[2]))
        color_tuple = (255,255,255)

        # find 4 points between lsho and rhip and 4 points between rsho and lhip to find something more precise
        seg_steps = [0.0, 0.25, 0.50, 0.75, 1.0]
        depths_torso = []
        poses_torso = []
        for step in seg_steps:

            p1 = step * lsho + (1 - step) * rhip
            if (
                p1[0] < depth_array.shape[1]
                and p1[1] < depth_array.shape[0]
            ):
                depth_p1 = depth_array[int(p1[1]), int(p1[0])]
                pose_p1 = self.pcl_array_xyz[
                    int(p1[1]), int(p1[0]), :
                ]
                if depth_p1 > 0:
                    depths_torso.append(depth_p1)
                    poses_torso.append(pose_p1)

            p2 = step * rsho + (1 - step) * lhip
            if (
                p2[0] < depth_array.shape[1]
                and p2[1] < depth_array.shape[0]
            ):
                depth_p2 = depth_array[int(p2[1]), int(p2[0])]
                pose_p2 = self.pcl_array_xyz[
                    int(p2[1]), int(p2[0]), :
                ]
                if depth_p2 > 0:
                    depths_torso.append(depth_p2)
                    poses_torso.append(pose_p2)

            if not self.args.no_show:
                # draw to check
                cv2.drawMarker(
                    self.vis_img,
                    (int(p1[0]), int(p1[1])),
                    color=color_tuple,
                    thickness=1,
                    markerType=cv2.MARKER_DIAMOND,
                    line_type=cv2.LINE_AA,
                    markerSize=8,
                )
                cv2.drawMarker(
                    self.vis_img,
                    (int(p2[0]), int(p2[1])),
                    color=color_tuple,
                    thickness=1,
                    markerType=cv2.MARKER_DIAMOND,
                    line_type=cv2.LINE_AA,
                    markerSize=8,
                )
        
        return depths_torso, poses_torso        
    
    @timeit
    def plot_body_pose_data(self, body_center, depth_body, pose_body, idx):

        color = RANDOM_COLORS[idx]
        # color_tuple = (int(color[0]), int(color[1]), int(color[2]))
        color_tuple = (255,255,255)

        cv2.drawMarker(
            self.vis_img,
            body_center,
            color = color_tuple,
            thickness=1,
            markerType=cv2.MARKER_TILTED_CROSS,
            line_type=cv2.LINE_AA,
            markerSize=16,
        )
        # cv2.putText(
        #     self.vis_img,
        #     "{:.0f}cm | {:.2f} {:.2f} {:.2f}".format(
        #         depth_body / 10,
        #         pose_body[0],
        #         pose_body[1],
        #         pose_body[2],
        #     ),
        #     (int(body_center[0]), int(body_center[1])),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.8 * TEXT_SCALE,
        #     (0, 255, 0),
        #     3,
        # )
        cv2.putText(
            self.vis_img,
            "{:.0f}cm".format(
                depth_body / 10
            ),
            (int(body_center[0]), int(body_center[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * TEXT_SCALE,
            (255, 255, 255),
            1,
        )

    @timeit
    def plot_skeleton_2d(self, keypoints, idx):

        color = RANDOM_COLORS[idx]
        # color_tuple = (int(color[0]), int(color[1]), int(color[2]))
        color_tuple = (255,255,255)

        for limb in COCO17_JOINTS_LIMBS:
            start = keypoints[limb[0], :]
            end = keypoints[limb[1], :]
            start_point = (int(start[0]), int(start[1]))
            end_point = (int(end[0]), int(end[1]))
            if (start[2] > self.args.kpt_thr) and (
                end[2] > self.args.kpt_thr
            ):
                cv2.line(
                    self.vis_img,
                    start_point,
                    end_point,
                    color = color_tuple,
                    thickness=1,
                )
    @timeit
    def plot_det_text_info(self, pose_closest):
        if pose_closest is not None:
            cv2.putText(
                self.vis_img,
                "{:.2f} {:.2f} {:.2f}".format(
                    pose_closest[0], pose_closest[1], pose_closest[2]
                ),
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2 * TEXT_SCALE,
                (255, 255, 255),
                5,
            )
            cv2.putText(
                self.vis_img,
                "{:.2f} {:.2f} {:.2f}".format(
                    pose_closest[0], pose_closest[1], pose_closest[2]
                ),
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2 * TEXT_SCALE,
                (0, 0, 0),
                3,
            )
        else:
            cv2.putText(
                self.vis_img,
                "No tracks with pose found",
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2 * TEXT_SCALE,
                (255, 255, 255),
                5,
            )
            cv2.putText(
                self.vis_img,
                "No tracks with pose found",
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2 * TEXT_SCALE,
                (0, 0, 0),
                3,
            )

    @timeit
    def plot_gaze_text_info(self, gaze_res, head_outputs, body_outputs, head_bb_abs, idx):
        prediction = gaze_res['direction']
        kappa = gaze_res['kappa'][0, -1].item()
        prediction_body = body_outputs['direction']
        prediction_head = head_outputs['direction']
        
        prediction_show = prediction.clone().cpu().detach().numpy()[0, -1, :]
        prediction_show_body = prediction_body.clone().cpu().detach().numpy()[0, -1, :]
        prediction_show_head = prediction_head.clone().cpu().detach().numpy()[0, -1, :]

        prediction_show_norm = prediction_show / np.linalg.norm(prediction_show)
        prediction_show_norm_body = prediction_show_body / np.linalg.norm(prediction_show_body)
        prediction_show_norm_head = prediction_show_head / np.linalg.norm(prediction_show_head)
        
        cv2.putText(
            self.vis_img,
            "Gaze {:.2f} {:.2f} {:.2f} ({:.2f})".format(
                prediction_show_norm[0], prediction_show_norm[1], prediction_show_norm[2], kappa
            ),
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1 * TEXT_SCALE,
            (255, 255, 255),
            5,
        )
        
        cv2.putText(
            self.vis_img,
            "Gaze {:.2f} {:.2f} {:.2f} ({:.2f})".format(
                prediction_show_norm[0], prediction_show_norm[1], prediction_show_norm[2], kappa
            ),
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1 * TEXT_SCALE,
            (0, 255, 0),
            3,
        )


    @timeit
    def plot_gaze_and_body_dir(self, gaze_res, head_outputs, body_outputs, head_bb_abs, body_bbox):
        head_bb_abs[2] += head_bb_abs[0]
        head_bb_abs[3] += head_bb_abs[1]
        
        prediction = gaze_res['direction']
        prediction_body = body_outputs['direction']
        prediction_head = head_outputs['direction']
        
        prediction_show = prediction.clone().cpu().detach().numpy()[0, -1, :]
        prediction_show_body = prediction_body.clone().cpu().detach().numpy()[0, -1, :]
        prediction_show_head = prediction_head.clone().cpu().detach().numpy()[0, -1, :]

        prediction_show_norm = prediction_show / np.linalg.norm(prediction_show)
        prediction_show_norm_body = prediction_show_body / np.linalg.norm(prediction_show_body)
        prediction_show_norm_head = prediction_show_head / np.linalg.norm(prediction_show_head)
        
        gaze_dir_2d = prediction_show_norm[0:2]
        body_dir_2d = prediction_show_norm_body[0:2]
        head_dir_2d = prediction_show_norm_head[0:2]
                
        body_center = (int((body_bbox[0] + body_bbox[2]) / 2), int((body_bbox[1] + body_bbox[3]) / 2))
        head_center = (int(head_bb_abs[0] / 2 + head_bb_abs[2] / 2), int(head_bb_abs[1] / 2 + head_bb_abs[3] / 2))

        des = (head_center[0] + int(gaze_dir_2d[0]*150), int(head_center[1] + gaze_dir_2d[1]*150))
        des_body = (body_center[0] + int(body_dir_2d[0]*150), int(body_center[1] + body_dir_2d[1]*150))
        des_head = (head_center[0] + int(head_dir_2d[0]*150), int(head_center[1] + head_dir_2d[1]*150))
  
        cv2.arrowedLine(self.vis_img, head_center, des, (0, 255, 0), 3, tipLength=0.3)        
        cv2.arrowedLine(self.vis_img, body_center, des_body, (0, 255, 255), 3, tipLength=0.3)        
        cv2.arrowedLine(self.vis_img, head_center, des_head, (255, 255, 255), 3, tipLength=0.3)        


    @timeit
    def plot_gaze_from_pitch_yaw(self, pitch, yaw, head_bb_abs, idx, keypoints):
        
        # color = RANDOM_COLORS[idx]
        # color_tuple = (int(color[0]), int(color[1]), int(color[2]))
        color_tuple = (0,0,255)

        head_bb_abs[2] += head_bb_abs[0]
        head_bb_abs[3] += head_bb_abs[1]
        
        prediction_show = np.zeros(3)
        prediction_show[0] = -np.sin(pitch) * np.cos(yaw)
        prediction_show[1] = -np.sin(yaw)
        prediction_show[2] = 999

        # prediction_show_norm = prediction_show / np.linalg.norm(prediction_show)

        gaze_dir_2d = prediction_show[0:2]
     
        # head_center = (int(head_bb_abs[0] / 2 + head_bb_abs[2] / 2), int(head_bb_abs[1] / 2 + head_bb_abs[3] / 2))
        head_center =  (int(keypoints[1,0] / 2 + keypoints[2,0] / 2), int(keypoints[1,1] / 2 + keypoints[2,1] / 2))
        
        des = (head_center[0] + int(gaze_dir_2d[0]*150), int(head_center[1] + gaze_dir_2d[1]*150))

        # cv2.arrowedLine(self.vis_img, head_center, des, (255,255,255), 3, tipLength=0.3)        
        cv2.arrowedLine(self.vis_img, head_center, des, color_tuple, 2, tipLength=0.1)        
        cv2.circle(self.vis_img, head_center, 5, color = color_tuple, thickness=-1)        

    @timeit
    def plot_gaze_angle_info(self, pitch, yaw, head_bb, idx):
        color = RANDOM_COLORS[idx]
        color_tuple = (int(color[0]), int(color[1]), int(color[2]))

        cv2.putText(
            self.vis_img,
            "{:.2f} {:.2f} deg".format(
                pitch, yaw
            ),
            (head_bb[0] + 30, head_bb[1] + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * TEXT_SCALE,
            (255,255,255),
            2,
        )
        cv2.putText(
            self.vis_img,
            "{:.2f} {:.2f} deg".format(
                pitch, yaw
            ),
            (head_bb[0] + 30, head_bb[1] + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * TEXT_SCALE,
            color_tuple,
            1,
        )
        
    @timeit
    def get_gafa_input_from_current_image(self, image, keypoints, body_yolo_bbox):

        body_yolo_bbox_int = {}
        body_yolo_bbox_int["u"] = int(body_yolo_bbox[0])
        body_yolo_bbox_int["v"] = int(body_yolo_bbox[1])
        body_yolo_bbox_int["w"] = int(body_yolo_bbox[2] - body_yolo_bbox[0])
        body_yolo_bbox_int["h"] = int(body_yolo_bbox[3] - body_yolo_bbox[1])

        # use torch instead of PIL because faster conversion
        # image_pil = PILImage.fromarray(image)
        image_torch = torch.from_numpy(image.copy()).moveaxis(2, 0)
                
        item = {
            "image": image_torch,
            "keypoints": keypoints[:, :2],
        }

        # get head bb in pixels
        head_trans = head_transform(item)
        head_bb = head_trans['bb']
        head_bb = np.array([head_bb['u'], head_bb['v'], head_bb['w'], head_bb['h']]).astype(np.float32)
        
        # get body bb in pixels
        # body_trans = body_transform(item) 
        body_trans = body_transform(item)
        body_bb = body_trans['bb']
        body_bb = np.array([body_bb['u'], body_bb['v'], body_bb['w'], body_bb['h']])
        body_image = body_trans['image'] # keep as tensor
        
        # change head bb to relative to body bb
        head_bb_abs = head_bb.copy()
        
        head_bb[0] -= body_bb[0]
        head_bb[1] -= body_bb[1]
        
        head_bb[0] = head_bb[0] / body_bb[2]
        head_bb[1] = head_bb[1] / body_bb[3]
        head_bb[2] = head_bb[2] / body_bb[2]
        head_bb[3] = head_bb[3] / body_bb[3]
                
        # store body center
        norm_body_center = (body_bb[[0, 1]] + body_bb[[2, 3]] / 2) / body_bb[[2,3]]
        
        # normalize image
        # img = normalize_img(image = body_image)['image'] # with albumnentations normalization
        # img = img.transpose(2, 0, 1) # with albumnentations normalization
        img = normalize_img_torch((body_image.float())/255) # ith torchvision normalization, to float and in range [0-1] before normalization

        assert(img.shape[0] == 3)
        assert(img.shape[1] == 256)
        assert(img.shape[2] == 192)
        
        # create mask of head bounding box
        head_mask = np.zeros((1, img.shape[1], img.shape[2]))
        head_bb_int = head_bb.copy()
        head_bb_int[[0, 2]] *= img.shape[2]
        head_bb_int[[1, 3]] *= img.shape[1]
        head_bb_int[2] += head_bb_int[0]
        head_bb_int[3] += head_bb_int[1]
        head_bb_int = head_bb_int.astype(np.int64)
        head_bb_int[head_bb_int < 0] = 0

        head_mask[:, head_bb_int[1]:head_bb_int[3], head_bb_int[0]:head_bb_int[2]] = 1
                        
        return img, head_mask, norm_body_center, head_bb_abs
    
    @timeit
    def plot_overlay_face_attention(self, track, head_bbox):
        x_min, y_min, x_max, y_max = map(int, head_bbox[:4])
        
        valid_depths = []
        valid_yaws = []
        valid_pitchs = []
        
        heat_count = 0
        history = 1
        length = min(len(track["depth_face"]), history)
        for i in range(length):
            index = len(track["depth_face"]) - i - 1
            yaw = np.rad2deg(track["gaze_yaw_rad"][index])
            pitch = np.rad2deg(track["gaze_pitch_rad"][index])
            depth = track["depth_face"][index]
            
            thresh = (int(depth / 1000) + 1) * 5 # 5 deg per meter
            if np.abs(yaw) < thresh and np.abs(pitch) < thresh:
                heat_count += 1
            
        cv2.putText(
            self.vis_img,
            "{:d}".format(
                heat_count
            ),
            (x_min - 30, y_min + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * TEXT_SCALE,
            (255,0,255),
            2,
        )
        
        overlay_img = self.vis_img.copy()
        cv2.rectangle(overlay_img, (x_min,y_min), (x_max,y_max), color = (0,255,0), thickness = -1)
        strength = (heat_count / history) * 0.75
        self.vis_img = cv2.addWeighted(self.vis_img,(1-strength),overlay_img,strength,0)

    @timeit
    def plot_overlay_face_attention_6d(self, track, head_bbox, keypoints):
        x_min, y_min, x_max, y_max = map(int, head_bbox[:4])
        
        valid_depths = []
        valid_yaws = []
        valid_pitchs = []
        
        heat_count = 0
        history = 20
        length = min(len(track["depth_face"]), history)
        for i in range(length):
            index = len(track["depth_face"]) - i - 1
            yaw = np.rad2deg(track["gaze_yaw_rad"][index])
            pitch = np.rad2deg(track["gaze_pitch_rad"][index])
            depth = track["depth_face"][index]
                        
            thresh = (int(depth / 1000) + 1) * 5 # 5 deg per meter
            if np.abs(yaw) < thresh and pitch < 0:
                heat_count += 1
                # suppose we are looking down

            
        # cv2.putText(
        #     self.vis_img,
        #     "{:d}".format(
        #         heat_count
        #     ),
        #     (x_min - 30, y_min + 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5 * TEXT_SCALE,
        #     (255,0,255),
        #     2,
        # )
        
        overlay_img = self.vis_img.copy()

        nose = keypoints[0,:2]        
        leye = keypoints[1,:2]
        reye = keypoints[2,:2]

        colorval = min(((heat_count * 2) / history), 1.0)
        strength = 0.5 + (heat_count / history) * 0.5 #(heat_count / history)
        cmap = get_mpl_colormap("Reds")
        color = (cmap[int(colorval * 255)] * 255)
        color_tuple = (int(color[0]), int(color[1]), int(color[2]))
                
        # radius = np.linalg.norm(reye - leye) / 4
        # cv2.circle(overlay_img, (int(leye[0]), int(leye[1])), int(radius), color = color_tuple, thickness = -1)
        # cv2.circle(self.vis_img, (int(leye[0]), int(leye[1])), int(radius), color = (0, 0, 0), thickness = 1)
        
        # cv2.circle(overlay_img, (int(reye[0]), int(reye[1])), int(radius), color = color_tuple, thickness = -1)
        # cv2.circle(self.vis_img, (int(reye[0]), int(reye[1])), int(radius), color = (0, 0, 0), thickness = 1)
        
        ellipse_center = (int(leye[0] / 2 + reye[0] / 2), int(leye[1] / 2 + reye[1] / 2))
        ellipse_height = int(nose[1] - (leye[1] / 2 + reye[1] / 2))
        ellipse_width = int((leye[0] - reye[0]) * 1.1)
        if ellipse_width > 0 and ellipse_height > 0:
            cv2.ellipse(overlay_img, ellipse_center, (ellipse_width, ellipse_height), 0, 0, 360, color_tuple, 3)
        
        # cv2.rectangle(overlay_img, (x_min,y_min), (x_max,y_max), color = (0,255,0), thickness = -1)
        # cv2.rectangle(self.vis_img, (x_min,y_min), (x_max,y_max), color = (255,255,255), thickness = 1)
        
        self.vis_img = cv2.addWeighted(self.vis_img,(1-strength),overlay_img,strength,0)
         

    def start(self):

        while not rospy.is_shutdown():

            if self.is_ready():

                image_count = self.current_image_count
                image_seq_unique = self.rgb_current_seq
                now = datetime.now()
                timestamp = now.strftime("%Y_%m_%d_%H_%M_%S_%f")

                if self.args.save or self.args.light_save:
                    self.save_rgb(image_count, image_seq_unique, timestamp)

                rgb_array = self.rgb.copy()

                if self.args.save:
                    self.save_depth(image_count, image_seq_unique, timestamp)

                depth_array = np.array(self.depth)
                depth_array[depth_array > self.depth_array_max_threshold] = (
                    self.depth_array_max_threshold
                )

                assert depth_array.shape[0] == rgb_array.shape[0]
                assert depth_array.shape[1] == rgb_array.shape[1]

                # Process RGB array
                if self.last_inferred_seq < self.rgb_current_seq:

                    current_frame_processing = self.rgb_current_seq
                    current_timestamp = self.rgb_timestamp
                    current_frame_id = self.rgb_frame_id
                    prInfo("Do inference on frame {}".format(current_frame_processing))

                    # keep old poses for tracking
                    pose_results_last = self.pose_results

                    tic = time.time()
                    mmdet_results = inference_detector(
                        self.det_model, rgb_array
                    )  # list of detection rectangle i.e [(x1,y1,x2,y2), ...]
                    tac = time.time()
                    prTimer("YOLO detection", tic, tac)
                    
                    # keep the person class bounding boxes.
                    person_results = process_mmdet_results(
                        mmdet_results, self.args.det_cat_id
                    )

                    new_persons = []
                    for person in person_results:
                        bbox = person["bbox"]
                        pt1 = (max(0, min(bbox[0], depth_array.shape[1]-1)), max(0,min(bbox[1], depth_array.shape[0]-1)) )
                        pt2 = (max(0, min(bbox[2], depth_array.shape[1]-1)), max(0,min(bbox[3], depth_array.shape[0]-1)) )
                        
                        # depth1 = depth_array[int(pt1[1]), int(pt1[0])]
                        # depth2 = depth_array[int(pt2[1]), int(pt2[0])]
                        # if depth1 > self.args.depth_limit_threshold or depth1 == 0 or depth2 > self.args.depth_limit_threshold or depth2 == 0:
                        #     pass
                        # else:
                        if abs(pt1[0] - pt2[0]) > self.args.bb_min_threshold/2 or abs(pt1[1]-pt2[1]) > self.args.bb_min_threshold:
                            new_persons.append(person)                            
                            
                    person_results = new_persons
                    
                    tic = time.time()
                    # test a single image, with a list of bboxes.
                    self.pose_results, returned_outputs = inference_top_down_pose_model(
                        self.pose_model,
                        rgb_array,
                        person_results,
                        bbox_thr=self.args.bbox_thr,
                        format="xyxy",
                        dataset=self.dataset,
                        dataset_info=self.dataset_info,
                        return_heatmap=self.return_heatmap,
                        outputs=None,
                    )
                    tac = time.time()
                    prTimer("ViTPose", tic, tac)
                    # get track id for each person instance
                    self.pose_results, self.next_id = get_track_id(
                        self.pose_results,
                        pose_results_last,
                        self.next_id,
                        use_oks=False,
                        tracking_thr=self.args.tracking_thr,
                        use_one_euro=self.args.euro,
                        fps=10,
                    )

                    # produce an output image
                    if not self.args.no_show:
                        self.vis_img = rgb_array.copy()

                    if self.display_all_detection and not self.args.no_show:
                        self.plot_mmdet_bbox(mmdet_results, depth_array.shape)

                    #### post processing, 3D lifting (if enabled) and gaze estimation (if enabled) ####

                    # remove too old tracks
                    for idx, track in list(self.tracks.items()):
                        if abs(image_count - track["last_seen"]) > self.args.max_frames_remove_tracks: 
                            prInfo("Removing track {}, not seen since frame {}, current is {}".format(idx, track["last_seen"], image_count))
                            self.tracks.pop(idx)
                            
                    self.tracks_in_current_image = {}

                    for res in self.pose_results:

                        # for each instance
                        bbox = res["bbox"]
                        keypoints = res["keypoints"]        
                        idx = res["track_id"] % 255
                        
                        if idx in self.tracks_in_current_image.keys():
                            prWarning("Track with idx {} (track_id {} from results) already in the current image, maybe because there are more than 255 detections in the image".format(
                                idx, res["track_id"]
                            ))
                            continue

                        if idx not in self.tracks.keys():
                            prInfo("Adding a new track with idx {}".format(idx))
                            self.tracks[idx] = {}
                            self.tracks[idx]["last_seen"] = image_count
                            self.tracks[idx]["keypoints_2d"] = []
                            self.tracks[idx]["images_crop"] = []
                            self.tracks[idx]["head_masks"] = []
                            self.tracks[idx]["norm_body_centers"] = []
                            self.tracks[idx]["bboxes"] = []
                            self.tracks[idx]["depth_face"] = []
                            self.tracks[idx]["gaze_yaw_rad"] = []
                            self.tracks[idx]["gaze_pitch_rad"] = []
                        
                        # add keypoint to the current track
                        self.tracks[idx]["last_seen"] = image_count
                        self.tracks[idx]["keypoints_2d"].append(keypoints)
                        self.tracks[idx]["bboxes"].append(bbox)
                        
                        self.tracks_in_current_image[idx] = {
                            "right_wrist_depth": None,
                            "right_wrist_pose": None,
                            "left_wrist_depth": None,
                            "left_wrist_pose": None,
                            "depth_center": None,
                            "pose_center": None,
                            "pose_from": None,
                            "depth_face": None,
                            "gaze_yaw_rad": None,
                            "gaze_pitch_rad": None,
                        }

                        # if history is long enough, process the trajectory with MotionBERT
                        if self.args.use_mb and len(self.tracks[idx]["keypoints_2d"]) >= self.args.mb_clip_len:
                            prInfo("Running MotionBERT for track {}".format(idx))
                            
                            # prepare motion
                            motion = np.asarray(self.tracks[idx]["keypoints_2d"]) # T, 17, 3
                            motion = motion[-self.args.mb_clip_len:, :, :] # keep only the required len
                            assert(motion.shape[1] == 17)
                            assert(motion.shape[2] == 3)
                            motion_h36 = coco2h36m(motion) # input is h36 format
                            motion_h36_scaled = crop_scale(motion_h36) # scale [1,1], normalize, crop
                            
                            with torch.no_grad():
                                current_input = torch.Tensor(motion_h36_scaled).unsqueeze(0).cuda()
                                tic = time.time()
                                predicted_3d_pos = self.motionbert_3d_model(current_input)
                                tac = time.time()
                                prTimer("MotionBERT", tic, tac)                  
                                # root relative
                                predicted_3d_pos[:,:,0,:] = 0  # [1,T,17,3]
                                
                                predicted_3d_pos_np = predicted_3d_pos[0,-1,:,:].cpu().numpy() # keep only the last prediction
                                if "keypoints_3d" in self.tracks[idx].keys():
                                    self.tracks[idx]["keypoints_3d"].append(predicted_3d_pos_np)
                                else:
                                    self.tracks[idx]["keypoints_3d"] = [predicted_3d_pos_np] * self.args.mb_clip_len # add fake padding at the begining so the lists align
                                
                                # print("len compare", idx, len(self.tracks[idx]["keypoints_3d"]), len(self.tracks[idx]["keypoints_2d"]), color = "yan")
                        
                        
                        # (run for every track or only closest ?) add input for gafa processing
                        # if for everyone should run in batch                                                
                        if self.args.use_gafa and (len(self.tracks[idx]["images_crop"]) >= self.args.gafa_n_frames or self.args.gafa_no_history):
                            gafa_tic = time.time()
                            
                            # Make sure that the image is rgb and not bgr, may need conversion !
                            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            # im_pil = Image.fromarray(img)
                            crop_img, head_mask, norm_body_center, head_bb_abs = self.get_gafa_input_from_current_image(rgb_array[:,:,::-1], keypoints, bbox)             

                            if self.args.gafa_no_history:
                                # no history : duplicate the last image
                                images = np.repeat(crop_img[np.newaxis, :, :, :], self.args.gafa_n_frames, axis = 0) # torch.Tensor of size [n_frames, 3, 256, 192]
                                head_masks = np.repeat(head_mask[np.newaxis, :, :, :], self.args.gafa_n_frames, axis = 0) # numpy.ndarray of size [n_frames, 3, 256, 192]
                                body_dvs = np.zeros((self.args.gafa_n_frames, 2)) # numpy.ndarray of size n_frames, 2
                                
                            else:
                                # history : use the last saved n images
                                self.tracks[idx]["images_crop"].append(crop_img)
                                self.tracks[idx]["head_masks"].append(head_mask)
                                self.tracks[idx]["norm_body_centers"].append(norm_body_center)
                            
                                images = torch.stack(self.tracks[idx]["images_crop"][-self.args.gafa_n_frames:], dim = 0) # torch.Tensor of size  n_frames, 3, 256, 192
                                head_masks = np.asarray(self.tracks[idx]["head_masks"][-self.args.gafa_n_frames:]) #  numpy.ndarray of size n_frames, 1, 256, 192
                                norm_body_centers = np.asarray(self.tracks[idx]["norm_body_centers"][-self.args.gafa_n_frames:]) # numpy.ndarray of size n_frames, 2
                                body_dvs = norm_body_centers - np.roll(norm_body_centers, shift=1, axis=0) # numpy.ndarray of size n_frames, 2
                                
                            with torch.no_grad():
                                # debug_dic = {}
                                
                                images = images.unsqueeze(0) #.cuda().float()              
                                head_masks = torch.from_numpy(head_masks).unsqueeze(0) #.cuda().float()              
                                body_dvs = torch.from_numpy(body_dvs).unsqueeze(0) #.cuda().float()     
                                
                                # last_img = images[0, -1, : ,: ,:].clone()
                                # for i in range(7):
                                #     images[0, i, : ,: ,:] = last_img
                                images = images.cuda().float()
                                                                
                                # last_mask = head_masks[0, -1, : ,: ,:].clone()
                                # for i in range(7):
                                #     head_masks[0, i, : ,: ,:] = last_mask
                                head_masks = head_masks.cuda().float()
                                
                                # body_dvs = torch.zeros(body_dvs.shape)
                                body_dvs = body_dvs.cuda().float()
                                                                
                                # debug_dic["images"] = images.clone().cpu().numpy()
                                # debug_dic["head_masks"] = head_masks.clone().cpu().numpy()
                                # debug_dic["body_dvs"] = body_dvs.clone().cpu().numpy()
                                
                                tic = time.time()
                                gaze_res, head_outputs, body_outputs = self.gafa_model(images, head_masks, body_dvs)
                                tac = time.time()
                                prTimer("GAFA", tic, tac)

                                # debug_dic["gaze_res"] = gaze_res["direction"].clone().cpu().numpy()
                                # debug_dic["head_outputs"] = head_outputs["direction"].clone().cpu().numpy()
                                # debug_dic["body_outputs"] = body_outputs["direction"].clone().cpu().numpy()
                                
                                # with open('./debug_dic.pickle', 'wb') as handle:
                                #     pickle.dump(debug_dic, handle)
                                
                                # print("GAFA done", color = "green", background = "white")
                                
                                if not self.args.no_show:
                                    self.plot_gaze_and_body_dir(gaze_res, head_outputs, body_outputs, head_bb_abs, bbox)
                                    self.plot_xyxy_person_bbox(idx, head_bb_abs, depth_array.shape)
                                    self.plot_gaze_text_info(gaze_res, head_outputs, body_outputs, head_bb_abs, idx)
                                
                                # For debug only
                                # plt.clf()
                                # plt.imshow(np.moveaxis(images.clone().cpu().numpy()[0,-1,:,:,:], 0, 2))
                                # plt.title("Custom data")
                                # plt.pause(0.01)

                            
                            gafa_tac = time.time()
                            prTimer("GAFA full", gafa_tic, gafa_tac)
                                                                            
                        else:
                            if self.args.use_gafa and self.args.gafa_no_history:
                                prInfo("Did not add inputs because no GAFA history required")
                            elif self.args.use_gafa:
                                # do not accumulate if unused
                                crop_img, head_mask, norm_body_center, head_bb_abs = self.get_gafa_input_from_current_image(rgb_array[:,:,::-1], keypoints, bbox)             
                                self.tracks[idx]["images_crop"].append(crop_img)
                                self.tracks[idx]["head_masks"].append(head_mask)
                                self.tracks[idx]["norm_body_centers"].append(norm_body_center)                        
                                prInfo("Didn't run GAFA yet, waiting for history")

                        
                        if self.args.use_gaze_resnet:
                            with torch.no_grad():
                                
                                # debug_dic = {"image_full": rgb_array}
                                
                                item = {"keypoints" : keypoints[:,:2]}
                                head_trans = head_transform_face(item)
                                head_bb = head_trans["bb"]
                                head_bb = np.array([head_bb['u'], head_bb['v'], head_bb['w'], head_bb['h']]).astype(np.int32)
                                
                                tic = time.time()
                                if self.args.use_face_detector:
                                    face_bboxes, fd_kp = self.face_detector.detect(rgb_array) # or convert to bgr ?? ## only use the body detection so that we match easily...
                                    prWarning("Using face_detector does not provide any matching to the current idx of the frame, only using first detection !")
                                else:
                                    face_bboxes = np.array([[head_bb[0],head_bb[1],head_bb[0]+head_bb[2],head_bb[1]+head_bb[3]]])
                                tac = time.time()
                                prTimer("Face detetction", tic, tac)
                                
                                if (face_bboxes.shape[0] > 0):
                                    
                                    x_min, y_min, x_max, y_max = map(int, face_bboxes[0,:4])
                                    head_image = rgb_array[y_min:y_max, x_min:x_max]
                                    
                                    if (head_image.shape[0] > 10) and (head_image.shape[1] > 10):
                                        head_image = pre_process(head_image)
                                    
                                        # For debug
                                        # plt.clf()
                                        # plt.imshow(np.moveaxis(head_image.clone().cpu().numpy()[0,:,:,:], 0, 2))
                                        # plt.title("Custom data")
                                        # plt.pause(0.01)

                                        # debug_dic["image"] = head_image
                                        
                                        pitch, yaw = self.gaze_estimation_model(head_image)

                                        # debug_dic["pitch"] = pitch
                                        # debug_dic["yaw"] = yaw
                                        
                                        # with open('debuig_dic.pkl', 'wb') as fp: 
                                        #     pickle.dump(debug_dic, fp)

                                        # Softmax beofre sum
                                        pitch_predicted, yaw_predicted = F.softmax(pitch, dim=1), F.softmax(yaw, dim=1)

                                        # Mapping from binned (0 to 90) to angles (-180 to 180) or (0 to 28) to angles (-42, 42)
                                        idx_tensor = torch.arange(90, device=self.args.device.lower(), dtype=torch.float32)
                                        
                                        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, dim=1) * 4 - 180
                                        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, dim=1) * 4 - 180
                                        
                                        pitch_predicted = pitch_predicted.cpu().numpy()
                                        yaw_predicted = yaw_predicted.cpu().numpy()
                                        
                                        # Degrees to Radians
                                        pitch_predicted_rad = np.radians(pitch_predicted)
                                        yaw_predicted_rad = np.radians(yaw_predicted)
                                        
                                        self.tracks_in_current_image[idx]["gaze_pitch_rad"] = pitch_predicted_rad
                                        self.tracks_in_current_image[idx]["gaze_yaw_rad"] = yaw_predicted_rad
                                        self.tracks[idx]["gaze_pitch_rad"].append(pitch_predicted_rad)
                                        self.tracks[idx]["gaze_yaw_rad"].append(yaw_predicted_rad)
                                           
                                        self.plot_gaze_from_pitch_yaw(pitch_predicted_rad[0], yaw_predicted_rad[0], head_bb, idx, keypoints)
                                        self.plot_gaze_angle_info(pitch_predicted[0], yaw_predicted[0], head_bb, idx)
                                                                            
                                        # get face depth
                                        nose = keypoints[0,:2].astype(np.uint32)
                                        leye = keypoints[1,:2].astype(np.uint32)
                                        reye = keypoints[2,:2].astype(np.uint32)
                                        
                                        depth_nose = depth_array[np.clip(nose[1], 0, depth_array.shape[0] - 1), np.clip(nose[0], 0, depth_array.shape[1] - 1)]
                                        depth_leye = depth_array[np.clip(leye[1], 0, depth_array.shape[0] - 1), np.clip(leye[0], 0, depth_array.shape[1] - 1)]
                                        depth_reye = depth_array[np.clip(reye[1], 0, depth_array.shape[0] - 1), np.clip(reye[0], 0, depth_array.shape[1] - 1)]
                                                                                
                                        depth_face = np.median([depth_nose, depth_leye, depth_reye])
                                                                                
                                        self.tracks_in_current_image[idx]["depth_face"] = depth_face
                                        self.tracks[idx]["depth_face"].append(depth_face)
                                                                                
                                        self.plot_overlay_face_attention(self.tracks[idx], face_bboxes[0,:4])
    
                                        

                        if self.args.use_six_d_rep:
                            with torch.no_grad():
                                
                                # debug_dic = {"image_full": rgb_array}
                                
                                item = {"keypoints" : keypoints[:,:2]}
                                head_trans = head_transform(item)
                                head_bb = head_trans["bb"]
                                head_bb = np.array([head_bb['u'], head_bb['v'], head_bb['w'], head_bb['h']]).astype(np.int32)
                                
                                tic = time.time()
                                if self.args.use_face_detector:
                                    face_bboxes = self.sixdrep_detector.detect(rgb_array, (640,640)) # or convert to bgr ?? ## only use the body detection so that we match easily...
                                    face_bboxes = face_bboxes.astype('int32')
                                    prWarning("Using face_detector does not provide any matching to the current idx of the frame, only using first detection !")
                                else:
                                    face_bboxes = np.array([[head_bb[0],head_bb[1],head_bb[0]+head_bb[2],head_bb[1]+head_bb[3]]])
                                tac = time.time()
                                prTimer("Face detetction", tic, tac)
                                                                
                                facing_camera = ((keypoints[3,0] - keypoints[4,0]) > 20)
                                
                                if (face_bboxes.shape[0] > 0) and facing_camera:
                                    x_min = face_bboxes[0,0]
                                    y_min = face_bboxes[0,1]
                                    x_max = face_bboxes[0,2]
                                    y_max = face_bboxes[0,3]
                                    box_w = abs(x_max - x_min)
                                    box_h = abs(y_max - y_min)

                                    x_min = max(0, x_min - int(0.2 * box_h))
                                    y_min = max(0, y_min - int(0.2 * box_w))
                                    x_max = x_max + int(0.2 * box_h)
                                    y_max = y_max + int(0.2 * box_w)

                                    head_image = rgb_array[y_min:y_max, x_min:x_max, :]
                                    
                                    if (head_image.shape[0] > 10) and (head_image.shape[1] > 10):

                                        
                                        head_image = PILImage.fromarray(head_image)
                                        head_image = head_image.convert('RGB')
                                        head_image = sixdreptransform(head_image)
                                        head_image = head_image.unsqueeze(0)

                                        head_image = head_image.cuda()
                                        head_image = head_image.half()

                                        tic = time.time()
                                        output = self.sixdrep_model(head_image)
                                        tac = time.time()
                                        prTimer("SixDRep", tic, tac)
                                                                        
                                        output = compute_euler(output) * 180 / np.pi

                                        p_output = output[:, 0].cpu()
                                        y_output = output[:, 1].cpu()
                                        r_output = output[:, 2].cpu()
                                                                        
                                        self.tracks_in_current_image[idx]["gaze_pitch_rad"] = np.deg2rad(p_output.item())
                                        self.tracks_in_current_image[idx]["gaze_yaw_rad"] = np.deg2rad(y_output.item())
                                        self.tracks[idx]["gaze_pitch_rad"].append(np.deg2rad(p_output.item()))
                                        self.tracks[idx]["gaze_yaw_rad"].append(np.deg2rad(y_output.item()))
                                           
                                        self.plot_gaze_from_pitch_yaw(np.deg2rad(y_output.item()), np.deg2rad(p_output.item()), head_bb, idx, keypoints) # invert pitch compared to resnet
                                        # self.plot_gaze_angle_info(y_output.item(), p_output.item(), head_bb, idx) # invert pitch compared to resnet
                                                                            
                                        # get face depth
                                        nose = keypoints[0,:2].astype(np.uint32)
                                        leye = keypoints[1,:2].astype(np.uint32)
                                        reye = keypoints[2,:2].astype(np.uint32)
                                        
                                        depth_nose = depth_array[np.clip(nose[1], 0, depth_array.shape[0] - 1), np.clip(nose[0], 0, depth_array.shape[1] - 1)]
                                        depth_leye = depth_array[np.clip(leye[1], 0, depth_array.shape[0] - 1), np.clip(leye[0], 0, depth_array.shape[1] - 1)]
                                        depth_reye = depth_array[np.clip(reye[1], 0, depth_array.shape[0] - 1), np.clip(reye[0], 0, depth_array.shape[1] - 1)]
                                                                                
                                        depth_face = np.median([depth_nose, depth_leye, depth_reye])
                                                                                
                                        self.tracks_in_current_image[idx]["depth_face"] = depth_face
                                        self.tracks[idx]["depth_face"].append(depth_face)
                                                                                
                                        self.plot_overlay_face_attention_6d(self.tracks[idx], face_bboxes[0,:4], keypoints)
                                    else:
                                        self.tracks[idx]["gaze_pitch_rad"].append(np.deg2rad(180))
                                        self.tracks[idx]["gaze_yaw_rad"].append(np.deg2rad(180))
                                        
                                        nose = keypoints[0,:2].astype(np.uint32)
                                        leye = keypoints[1,:2].astype(np.uint32)
                                        reye = keypoints[2,:2].astype(np.uint32)
                            
                                        depth_nose = depth_array[np.clip(nose[1], 0, depth_array.shape[0] - 1), np.clip(nose[0], 0, depth_array.shape[1] - 1)]
                                        depth_leye = depth_array[np.clip(leye[1], 0, depth_array.shape[0] - 1), np.clip(leye[0], 0, depth_array.shape[1] - 1)]
                                        depth_reye = depth_array[np.clip(reye[1], 0, depth_array.shape[0] - 1), np.clip(reye[0], 0, depth_array.shape[1] - 1)]
                                        depth_face = np.median([depth_nose, depth_leye, depth_reye])

                                        self.tracks[idx]["depth_face"].append(depth_face)
                                
                                else:
                                    self.tracks[idx]["gaze_pitch_rad"].append(np.deg2rad(180))
                                    self.tracks[idx]["gaze_yaw_rad"].append(np.deg2rad(180))
                                    
                                    nose = keypoints[0,:2].astype(np.uint32)
                                    leye = keypoints[1,:2].astype(np.uint32)
                                    reye = keypoints[2,:2].astype(np.uint32)
                        
                                    depth_nose = depth_array[np.clip(nose[1], 0, depth_array.shape[0] - 1), np.clip(nose[0], 0, depth_array.shape[1] - 1)]
                                    depth_leye = depth_array[np.clip(leye[1], 0, depth_array.shape[0] - 1), np.clip(leye[0], 0, depth_array.shape[1] - 1)]
                                    depth_reye = depth_array[np.clip(reye[1], 0, depth_array.shape[0] - 1), np.clip(reye[0], 0, depth_array.shape[1] - 1)]
                                    depth_face = np.median([depth_nose, depth_leye, depth_reye])

                                    self.tracks[idx]["depth_face"].append(depth_face)
                    
                        # Draw bb              
                        bbox[4] *= 100
                        bbox = bbox.astype(np.int32)
                        
                        if not self.args.no_show:
                            self.plot_xyxy_person_bbox(idx, bbox, depth_array.shape, self.tracks[idx])

                        # return the list of body center joints and also fill self.tracks_in_current_image[idx]
                        body_center_joints = self.process_keypoints(keypoints, depth_array, idx)

                        # find the body center
                        if len(body_center_joints) == 4:
                            # if we managed to find the 4 points of the torso, search on the torso
                            body_center_joints = np.array(
                                body_center_joints
                            )  # lsho, rsho, lhip, rhip
                            lsho = body_center_joints[0, :]
                            rsho = body_center_joints[1, :]
                            lhip = body_center_joints[2, :]
                            rhip = body_center_joints[3, :]

                            depths_torso, poses_torso = self.get_depth_and_poses_of_torso(depth_array, lsho, rsho, lhip, rhip, idx)

                            # redraw bb with more info
                            if not self.args.no_show:
                                self.plot_xyxy_person_bbox(idx, bbox, depth_array.shape, self.tracks[idx], poses_torso)

                            if len(depths_torso) > 3:
                                # at least 4 points to average decently
                                depth_body = np.array(depths_torso).mean()
                                pose_body = np.array(poses_torso).mean(axis=0)
                                self.tracks_in_current_image[idx][
                                    "depth_center"
                                ] = depth_body  # mm
                                self.tracks_in_current_image[idx][
                                    "pose_center"
                                ] = pose_body  # m
                                self.tracks_in_current_image[idx]["pose_from"] = "torso"

                                # just for drawing
                                body_center = np.mean(body_center_joints, axis=0)
                                # Draw center of body
                                body_center = (int(body_center[0]), int(body_center[1]))
                                
                                if not self.light_display and not self.args.no_show:
                                    self.plot_body_pose_data(body_center, depth_body, pose_body, idx)

                        else:
                            # if we did not managed to find the 4 points of the torso, search in the bbox
                            prWarning(
                                "Can't use body center from shoulders and hips for track {} : do nothing".format(
                                    idx
                                )
                            )

                        # draw skeleton
                        if not self.args.no_show and not self.args.light_display:
                            self.plot_skeleton_2d(keypoints, idx)

                    min_depth = 1e6  # mm
                    min_depth_idx = -1
                    for idx, track_info in self.tracks_in_current_image.items():
                        depth = track_info["depth_center"]
                        if depth is not None:
                            if depth < min_depth:
                                min_depth = depth
                                min_depth_idx = idx

                    if min_depth_idx != -1:
                        pose_closest = self.tracks_in_current_image[min_depth_idx][
                            "pose_center"
                        ]
                        yaw_closest_gaze = self.tracks_in_current_image[min_depth_idx]["gaze_yaw_rad"]
                        if yaw_closest_gaze is None:
                            yaw_closest = np.deg2rad(-180.0)
                        else:
                            yaw_closest = yaw_closest_gaze
                        prInfo(
                            "Using track {} as it is the closest".format(min_depth_idx)
                        )
                        tf_msg = TransformStamped()
                        tf_msg.child_frame_id = args.namespace + "/human"
                        tf_msg.header.seq = current_frame_processing
                        tf_msg.header.stamp = current_timestamp
                        tf_msg.header.frame_id = current_frame_id
                        # adapt to robot camera frame convention on the robot
                        tf_msg.transform.translation.x = pose_closest[2]
                        tf_msg.transform.translation.y = -pose_closest[0]
                        tf_msg.transform.translation.z = -pose_closest[1]

                        angle = np.arctan(
                            tf_msg.transform.translation.y
                            / tf_msg.transform.translation.x
                        )

                        # Rotate to have 'human' x axis looking towards the robot
                        rot = Rotation()
                        rot.DoRotZ(angle)
                        rot.DoRotY(np.pi)
                        qx, qy, qz, qw = rot.GetQuaternion()

                        tf_msg.transform.rotation.x = qx
                        tf_msg.transform.rotation.y = qy
                        tf_msg.transform.rotation.z = qz
                        tf_msg.transform.rotation.w = qw


                        dist = np.sqrt(
                            tf_msg.transform.translation.x**2 + tf_msg.transform.translation.y**2 + tf_msg.transform.translation.z**2
                        )

                        if dist < self.args.max_distance: # meters
                            self.goal_pub.publish(tf_msg)
                            prSuccess(
                                "Publishing coordinates {:.2f} {:.2f} {:.2f}".format(
                                    pose_closest[0], pose_closest[1], pose_closest[2]
                                )
                            )

                            self.tf_br.sendTransform(tf_msg)

                        prSuccess(
                            "Publishing coordinates {:.2f} {:.2f} {:.2f} and yaw {:.2f}".format(
                                pose_closest[0], pose_closest[1], pose_closest[2], np.rad2deg(yaw_closest)
                            )
                        )

                        self.tf_br.sendTransform(tf_msg)

                        if not self.args.no_show:
                            # self.plot_det_text_info(pose_closest)
                            pass

                    else:
                        
                        if not self.args.no_show:
                            # self.plot_det_text_info(None)
                            pass

                    self.last_inferred_seq = current_frame_processing

                    if self.args.save and not self.args.no_show:
                        self.save_output_image(image_count, image_seq_unique, timestamp)

                else:
                    prWarning(
                        "No inference because the current RGB frame has already been processed last_inferred_seq {} vs rgb_current_seq {}".format(
                            self.last_inferred_seq, self.rgb_current_seq
                        )
                    )

                if not self.args.no_show:
                    depth_array_disp = depth_array.copy()
                    depth_array_disp[depth_array_disp > 3000] = 3000
                    depth_array_norm = ((depth_array_disp - depth_array_disp.min())) / (
                        depth_array_disp.max() - depth_array_disp.min()
                    )
                    depth_array_norm = depth_array_norm * 255
                    depth_array_norm = depth_array_norm.astype(np.uint8)
                    depth_array_norm_colored = (
                        self.depth_cmap[depth_array_norm] * 255
                    ).astype(np.uint8)

                if self.args.save or self.args.light_save:
                    self.save_pcl(image_count, image_seq_unique, timestamp)

                if self.vis_img is not None:
                    full_display_array = np.zeros(
                        (rgb_array.shape[0] * 2, rgb_array.shape[1], 3), dtype=np.uint8
                    )
                    full_display_array[: rgb_array.shape[0], :, :] = self.vis_img
                    full_display_array[rgb_array.shape[0] :, :, :] = (
                        depth_array_norm_colored
                    )

                    if not self.args.no_show:
                        cv2.imshow("RGBD window", full_display_array)
                        cv2.waitKey(3)

            else:
                print("Images are None !")

            self.loop_rate.sleep()


if __name__ == "__main__":

    ## Parser with params
    parser = ArgumentParser()
    parser.add_argument(
        "--det_config",
        type=str,
        default="./configs/detection/yolov3_d53_320_273e_coco.py",
        help="Config file for detection | default = %(default)s",
    )
    parser.add_argument(
        "--det_checkpoint",
        type=str,
        default="./models/yolov3_d53_320_273e_coco-421362b6.pth",
        help="Checkpoint file for detection | default = %(default)s",
    )
    parser.add_argument(
        "--pose_config",
        type=str,
        default="./configs/pose/ViTPose_small_coco_256x192.py",
        help="Config file for pose | default = %(default)s",
    )
    parser.add_argument(
        "--pose_checkpoint",
        type=str,
        default="./models/vitpose_small.pth",
        help="Checkpoint file for pose | default = %(default)s",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device used for inference | default = %(default)s",
    )
    parser.add_argument(
        "--det_cat_id",
        type=int,
        default=1,
        help="Category id for bounding box detection model (person) | default = %(default)s",
    )
    parser.add_argument(
        "--bbox_thr",
        type=float,
        default=0.3,
        help="Bounding box score threshold | default = %(default)s",
    )
    parser.add_argument(
        "--kpt_thr",
        type=float,
        default=0.3,
        help="Keypoint score threshold | default = %(default)s",
    )
    parser.add_argument(
        "--tracking_thr",
        type=float,
        default=0.3,
        help="Tracking threshold | default = %(default)s",
    )
    parser.add_argument(
        "--euro", action="store_true", help="Using One_Euro_Filter for smoothing"
    )
    # parser.add_argument('--rgb_topic', default = "orbbec/rgb", type=str, help='ROS topic for RGB image')
    # parser.add_argument('--depth_topic', default = "orbbec/depth", type=str, help='ROS topic for depth image')
    # parser.add_argument('--pcl_topic', default = "orbbec/pcl", type=str, help='ROS topic for pcl')
    parser.add_argument(
        "--namespace",
        default="orbbec",
        type=str,
        help="ROS topic namespace for rgb, depth, pcl | default = %(default)s",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        default=False,
        help="whether to show visualizations | default = %(default)s",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="whether to save images (rgb and d and predictions and pcl) | default = %(default)s",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        default=False,
        help="whether to flip images | default = %(default)s",
    )
    parser.add_argument(
        "--light_save",
        action="store_true",
        default=False,
        help="whether to save only rgb and pcl (not optimized use the light_save of visualizer for optimized saving) | default = %(default)s",
    )
    parser.add_argument(
        "--display_all_detection",
        "-dad",
        action="store_true",
        default=False,
        help="whether to display all detections or only human | default = %(default)s",
    )
    parser.add_argument(
        "--light_display",
        "-ld",
        action="store_true",
        default=False,
        help="whether to display only skeletons | default = %(default)s",
    )
    parser.add_argument("--fps", type=int, default=10, help="Node and recording fps")
    parser.add_argument(
        "--depth_cmap",
        default="jet",
        type=str,
        help="mpl colormap for depth image | default = %(default)s",
    )

    parser.add_argument('--mb_3d_config', type=str, default = "./configs/pose3d/MB_ft_h36m.yaml", help='Config file for 3D poses | default = %(default)s')
    parser.add_argument('--mb_checkpoint', type=str, default = "./checkpoint/pose3d/MB_train_h36m/best_epoch.bin", help='Checkpoint file for 3D poses | default = %(default)s')
    parser.add_argument(
        '--mb_clip_len',
        type=int,
        default=10,
        help='Number of past frames to use for MotionBERT (default in model is 243) | default = %(default)s')
    parser.add_argument(
        '--max_frames_remove_tracks',
        type=int,
        default=2,
        help='Number frames without the track present to keep going before removing a track | default = %(default)s')
    parser.add_argument(
        "--use_mb",
        "-mb",
        action="store_true",
        default=False,
        help="whether to use MotionBERT 3D Lifter | default = %(default)s",
    )

    parser.add_argument('--gafa_checkpoint', type=str, default = "./checkpoint/gafa/GazeNet_PyTorch.pt", help='Checkpoint file for 3D gaze estimation GAFA | default = %(default)s')
    parser.add_argument(
        '--gafa_n_frames',
        type=int,
        default=7,
        help='Number of past frames to use for GAFA (default in model is 7) | default = %(default)s')
    parser.add_argument(
        "--use_gafa",
        "-gafa",
        action="store_true",
        default=False,
        help="whether to use GAFA 3D Gaze Estimation | default = %(default)s",
    )
    parser.add_argument(
        "--gafa_no_history",
        "-gnh",
        action="store_true",
        default=False,
        help="whether to use history in the GAFA sequence or fake it by copying last image | default = %(default)s",
    )

    parser.add_argument(
        "--use_gaze_resnet",
        "-resnet",
        action="store_true",
        default=False,
        help="whether to use Gaze ResNet18 3D Gaze Estimation | default = %(default)s",
    )
    parser.add_argument(
        "--use_face_detector",
        "-ufd",
        action="store_true",
        default=False,
        help="whether to use Face Detector before gaze ResNet18 3D Gaze Estimation, or juste use bbox from keypoints | default = %(default)s",
    )
    parser.add_argument(
        "--use_six_d_rep",
        "-sixdrep",
        action="store_true",
        default=False,
        help="whether to use 6D rep head pose estimation instead of gaze estimation | default = %(default)s",
    )
    parser.add_argument(
        "--bb_min_threshold",
        "-bbmt",
        type=int,
        default=0,
        help="Minimum height of bb in pixels | default = %(default)s",
    )
    parser.add_argument(
        "--max_distance",
        type=float,
        default=2.5,
        help="Maximum distance allowed for publishing human pose | default = %(default)s",
    )
    
    args = parser.parse_args()

    assert has_mmdet, "Please install mmdet to run the demo."
    assert args.det_config is not None
    assert args.det_checkpoint is not None
    
    if args.use_mb:
        assert(has_mb), "Option --use_mb requires MotionBERT install"

    if args.use_gafa:
        assert(args.use_gaze_resnet == False), "Option --use_gafa and --use_gaze_resnet are not compatible"
        assert(args.use_six_d_rep == False), "Option --use_gafa and --use_six_d_rep are not compatible"

    if args.use_gaze_resnet:
        assert(args.use_gafa == False), "Option --use_gaze_resnet and --use_gafa are not compatible"
        assert(args.use_six_d_rep == False), "Option --use_gaze_resnet and --use_six_d_rep are not compatible"

    if args.use_six_d_rep:
        assert(args.use_gaze_resnet == False), "Option --use_six_d_rep and --use_gaze_resnet are not compatible"
        assert(args.use_gafa == False), "Option --use_six_d_rep and --use_gafa are not compatible"

    if args.use_gafa:
        assert(has_gafa), "Option --use_gafa requires GAFA install"

    if args.use_six_d_rep:
        assert(has_sixdrep),  "Option --use_six_d_rep requires 6D Rep install"
    
    if args.use_gaze_resnet:
        assert(has_gaze_est), "Option --use_gaze_resnet requires Gaze Estimation"
    
    prInfo("Loaded with args : {}".format(args))

    rospy.init_node("python_orbbec_inference", anonymous=True)
    my_node = InferenceNodeRGBD(args)
    my_node.start()
    cv2.destroyAllWindows()
