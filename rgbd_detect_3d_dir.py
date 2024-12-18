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
from PIL import Image as PILImage

# custom utils import
from utils import *

# torch import
import torch

# transform for bounding boxes from keypoints
from transform_utils import head_transform
    
# 6D Rep import
try:
    has_sixdrep = True
    from sixdrep.util import FaceDetector, compute_euler
    from sixdrep.utils import sixdreptransform
except:
    has_sixdrep = False
    prWarning("No 6D Rep import, fail")
    
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
        
        # if enabled, init 6DRep
        if self.args.use_six_d_rep:
            
            self.sixdrep_model = torch.load(f=args.six_d_rep_checkpoint, map_location='cuda')
            self.sixdrep_model = self.sixdrep_model['model'].float().fuse()
            
            if self.args.use_face_detector:
                self.sixdrep_detector = FaceDetector(args.six_d_rep_face_detector_checkpoint)
            else:
                self.sixdrep_detector = None
                
            self.sixdrep_model.half()
            self.sixdrep_model.eval()

        
        # dataset object for detection and pose from MMPose
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
        
        # shared variables for the received images and pcl
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
        
        # counter for the incoming frames
        self.pcl_current_seq = -1
        self.rgb_current_seq = -1
        self.last_inferred_seq = -1
        self.depth_current_seq = -1
        self.current_image_count = 0
        self.rgb_frame_id = None # received from ROS image, used as frame_id in publisher

        # CV Bridge for receiving frames
        self.br = CvBridge()

        # Set ROS node rate
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

        if args.save:
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
        """ Make sure the node is ready to process (i.e. it has received an RGB image, a depth image and a PCL)

        Returns:
            bool: True if ready
        """
        ready = (
            (self.rgb is not None)
            and (self.depth is not None)
            and (self.pcl_array_xyz is not None)
        )
        return ready
    
    @timeit
    def save_rgb(self, image_count, image_seq_unique, timestamp):
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
        """ Plot all the bounding box detected in the image by the mmdet model (eg. YOLOv3) if score is superior to bbox_thr. Add classes names (assuming it is from YOLO_COCO_80_CLASSES).

        Args:
            mmdet_results (list): list of detections following the format of mmdet output
            array_shape (np.array): array describing the output image width and height
        """
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
        """ Plot the bounding box of detected humans with associated informations

        Args:
            idx (int): track if of the detected human
            bbox (np.array or list): bounding box of the detected human [x_min, y_min, x_max, y_max, score]
            array_shape (np.array or list): array describing the output image width and height
            track (dict): current track (human) information, may include gaze orientation information
            poses_torso (np.ndarray, optional): position of the body center in camera frame. Defaults to None.
        """
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
        
        # yolo score
        score = bbox[4]
        
        # current gaze
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
            
        # attention
        heat_count = 0
        history = 20
        length = min(len(track["depth_face"]), history)
        for i in range(length):
            index = len(track["depth_face"]) - i - 1
            yaw = np.rad2deg(track["gaze_yaw_rad"][index])
            pitch = np.rad2deg(track["gaze_pitch_rad"][index])
            depth = track["depth_face"][index]
                        
            thresh = (int(depth / 1000) + 1) * 6 # 6 deg per meter # TODO : set attention condition somewhere else
            if np.abs(yaw) < thresh and pitch < 0:
                # assume we are looking down
                heat_count += 1
        
        attention_score = int( min((heat_count * 2 / history), 1) * 100)
        
        draw_bbox_with_corners(self.vis_img, bbox_ints, color = color_tuple, thickness = 5, proportion = 0.2)
        
        text = "ID : {} | Person : {}% | Attention : {}%".format(idx, score, attention_score)
        if poses_torso is not None and  type(pose_body) == np.ndarray:
            text2 = "Yaw = {} | Pitch = {} | pos = ({:.2f}, {:.2f}, {:.2f})".format(yaw_g, pitch_g, pose_body_n[0], pose_body_n[1], pose_body_n[2])
        else:
            text2 = "Pose undefined (shoulder and/or hips not visible) | Yaw = {} | Pitch = {}".format(yaw_g, pitch_g)
        
        cv2.putText(
            self.vis_img,
            text,
            (bbox_ints[0], bbox_ints[1] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * TEXT_SCALE,
            color_tuple,
            1,
        )
        
        # if poses_torso is not None and type(pose_body) == np.ndarray:
        cv2.putText(
            self.vis_img,
            text2,
            (bbox_ints[0], bbox_ints[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35 * TEXT_SCALE,
            color_tuple,
            1,
        )
             
    @timeit   
    def process_keypoints(self, keypoints, depth_array, idx):
        """Iterate over all keypoints detected for the given track and process them, add special keypoints (wrists) to the track informations in self.tracks_in_current_image

        Args:
            keypoints (np.ndarray): array of keypoints detected in the image shape is expected to be ([17,3] : COCO17 format with x,y pixel coordinates + score)
            depth_array (np.ndarray): array of depth calibrated with the RGB image (depth in mm)
            idx (int): id of the current track

        Returns:
            List: list of body center joints (arrays of size [3] : x,y pixel coordinates + score), i.e. hips and shoulders, only if they are detected with a score > kpt_thr and inside the frame
        """
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

                if not self.args.no_show:
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
                    if not self.args.no_show:
                        cv2.drawMarker(
                            self.vis_img,
                            (int(kp[0]), int(kp[1])),
                            color=color_tuple,
                            thickness=3,
                            markerType=cv2.MARKER_CROSS,
                            line_type=cv2.LINE_AA,
                            markerSize=8,
                        )

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
                    if not self.args.no_show:
                        cv2.drawMarker(
                            self.vis_img,
                            (int(kp[0]), int(kp[1])),
                            color=color_tuple,
                            thickness=3,
                            markerType=cv2.MARKER_CROSS,
                            line_type=cv2.LINE_AA,
                            markerSize=8,
                        )
                        
        return body_center_joints

    @timeit
    def get_depth_and_poses_of_torso(self, depth_array, lsho, rsho, lhip, rhip, idx):
        """ Compute the depth and position of the detected human using multiple points on its torso

        Args:
            depth_array (np.ndarray): depth image calibrated with RGB image (in mm)
            lsho (np.array): array of size [3] with left shoulder keypoint data : x,y pixel coordinates and score 
            rsho (np.array): array of size [3] with right shoulder keypoint data : x,y pixel coordinates and score 
            lhip (np.array): array of size [3] with left hip keypoint data : x,y pixel coordinates and score 
            rhip (np.array): array of size [3] with right hip keypoint data : x,y pixel coordinates and score 
            idx (int): id of the current description

        Returns:
            List, List: lists of valid depth and poses (if depth is 0, ie. pixel is a blank spot in the depth image, then it is discarded)
        """

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
        """Plot skeleton limbds assuming the keypoints are COCO17 format

        Args:
            keypoints (np.ndarray): array of shape [17,3] with the body keypoints 
            idx (int): id of the track
        """
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
    def plot_gaze_from_pitch_yaw(self, pitch, yaw, head_bb_abs, idx, keypoints):
        """ Plot 2D reprojection of the computed gaze direction

        Args:
            pitch (float): pitch of gaze in rad
            yaw (float): yaw of gaze in rad (0 = facing camera)
            head_bb_abs (lsit of np.array): bounding box of the head with format [x_min, y_min, width, height]
            idx (int): id of the current track
            keypoints (np.ndarray): array of shape [17,3] with the body keypoints 
        """
        
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
    def plot_overlay_face_attention_6d(self, track, head_bbox, keypoints):
        """ Plot ellipse around the eyes, colored depending on the attention of the person

        Args:
            track (dict): information oin the current track including yaw and pitch of gaze in the last frames
            head_bbox (np.array): array of size [4] or [5] with bounding box as [x_min, y_min, x_max, y_max, (score)]
            keypoints (np.ndarray): array of shape [17,3] with the body keypoints 
        """
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

        
        overlay_img = self.vis_img.copy()

        nose = keypoints[0,:2]        
        leye = keypoints[1,:2]
        reye = keypoints[2,:2]

        colorval = min(((heat_count * 2) / history), 1.0)
        strength = 0.5 + (heat_count / history) * 0.5 #(heat_count / history)
        cmap = get_mpl_colormap("Reds")
        color = (cmap[int(colorval * 255)] * 255)
        color_tuple = (int(color[0]), int(color[1]), int(color[2])) 

        
        ellipse_center = (int(leye[0] / 2 + reye[0] / 2), int(leye[1] / 2 + reye[1] / 2))
        ellipse_height = int(nose[1] - (leye[1] / 2 + reye[1] / 2))
        ellipse_width = int((leye[0] - reye[0]) * 1.1)
        if ellipse_width > 0 and ellipse_height > 0:
            cv2.ellipse(overlay_img, ellipse_center, (ellipse_width, ellipse_height), 0, 0, 360, color_tuple, 3)
        
        self.vis_img = cv2.addWeighted(self.vis_img,(1-strength),overlay_img,strength,0)
         

    def start(self):

        while not rospy.is_shutdown():

            if self.is_ready():

                image_count = self.current_image_count
                image_seq_unique = self.rgb_current_seq
                now = datetime.now()
                timestamp = now.strftime("%Y_%m_%d_%H_%M_%S_%f")

                if self.args.save:
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
                        
                        if abs(pt1[0] - pt2[0]) > self.args.bb_min_threshold/2 or abs(pt1[1]-pt2[1]) > self.args.bb_min_threshold:
                            # Filter bbox if they are too small
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
                        use_one_euro=False,
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

                        if self.args.use_six_d_rep:
                            with torch.no_grad():
                                
                                # debug_dic = {"image_full": rgb_array}
                                
                                item = {"keypoints" : keypoints[:,:2]}
                                head_trans = head_transform(item)
                                head_bb = head_trans["bb"]
                                head_bb = np.array([head_bb['u'], head_bb['v'], head_bb['w'], head_bb['h']]).astype(np.int32)
                                
                                tic = time.time()
                                if self.args.use_face_detector:
                                    x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                                    x_min = np.clip(x_min, 0, rgb_array.shape[1] - 1)
                                    y_min = np.clip(y_min, 0, rgb_array.shape[0] - 1)
                                    x_max = np.clip(x_max, 0, rgb_array.shape[1] - 1)
                                    y_max = np.clip(y_max, 0, rgb_array.shape[0] - 1)
                
                                    if (x_max-x_min) > 10 and (y_max-y_min) > 10:
                                        body_rgb = rgb_array[y_min:y_max, x_min:x_max, :]
                                        face_bboxes = self.sixdrep_detector.detect(body_rgb, (640,640)) # or convert to bgr ?? ## only use the body detection so that we match easily...
                                        face_bboxes = face_bboxes.astype('int32')

                                        face_bboxes[:,0] += x_min
                                        face_bboxes[:,1] += y_min
                                        face_bboxes[:,2] += x_min
                                        face_bboxes[:,3] += y_min
                                        if face_bboxes.shape[0] > 1:
                                            prWarning("More than one face detected in the bounding box of track {}, only using first detection !".format(idx))
                                        elif face_bboxes.shape[0] == 1:
                                            draw_bbox_with_corners(self.vis_img, face_bboxes[0,:])
                                    else:
                                        face_bboxes = np.asarray([])
                                else:
                                    face_bboxes = np.array([[head_bb[0],head_bb[1],head_bb[0]+head_bb[2],head_bb[1]+head_bb[3]]])
                                tac = time.time()
                                prTimer("Face detetction", tic, tac)
                                                                
                                facing_camera = ((keypoints[3,0] - keypoints[4,0]) > 20) # heuristic to check if approximetly facing the camera # TODO : add number of pixels as parameter
                                
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
                    
                        # Draw bb              
                        bbox[4] *= 100 # score percentage
                        bbox = bbox.astype(np.int32)
                        
                        if not self.args.no_show:
                            self.plot_xyxy_person_bbox(idx, bbox, depth_array.shape, self.tracks[idx])

                        # return the list of body center joints and also fill self.tracks_in_current_image[idx]
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
                                
                                if not self.args.no_show:
                                    self.plot_body_pose_data(body_center, depth_body, pose_body, idx)

                        else:
                            # if we did not managed to find the 4 points of the torso, search in the bbox
                            prWarning(
                                "Can't use body center from shoulders and hips for track {} : do nothing".format(
                                    idx
                                )
                            )

                        # draw skeleton
                        if not self.args.no_show:
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

                if self.args.save:
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
        help="whether to save images (rgb and d and predictions and pcl), warning : using this option significantly slows down the process | default = %(default)s",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        default=False,
        help="whether to flip images | default = %(default)s",
    )
    parser.add_argument(
        "--display_all_detection",
        "-dad",
        action="store_true",
        default=False,
        help="whether to display all detections or only human | default = %(default)s",
    )
    parser.add_argument(
        "--fps", 
        type=int, 
        default=10, 
        help="Node and recording fps"
    )
    parser.add_argument(
        "--depth_cmap",
        default="jet",
        type=str,
        help="mpl colormap for depth image | default = %(default)s",
    )
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
        "--use_face_detector",
        "-ufd",
        action="store_true",
        default=False,
        help="whether to use Face Detector 3D Gaze or Head Pose Estimation, or juste use bbox from keypoints / better but slower especially with multiple people around | default = %(default)s",
    )
    parser.add_argument(
        "--use_six_d_rep",
        "-sixdrep",
        action="store_true",
        default=False,
        help="whether to use 6D rep head pose estimation instead of gaze estimation | default = %(default)s",
    )
    parser.add_argument(
        "--six_d_rep_checkpoint",
        type=str,
        default="./models/sixdrep_best.pt",
        help="Config file for head pose estimation | default = %(default)s",
    )
    parser.add_argument(
        "--six_d_rep_face_detector_checkpoint",
        type=str,
        default="./models/sixdrep_detection.onnx",
        help="Checkpoint file for detection | default = %(default)s",
    )
    parser.add_argument(
        "--bb_min_threshold",
        "-bbmt",
        type=int,
        default=0,
        help="Minimum height of bb in pixels | default = %(default)s",
    )
    
    args = parser.parse_args()

    assert has_mmdet, "mmdet install error."
    assert args.det_config is not None
    assert args.det_checkpoint is not None
    
    if args.use_six_d_rep:
        assert(has_sixdrep),  "Option --use_six_d_rep requires 6D Rep install"
    
    prInfo("Loaded with args : {}".format(args))

    rospy.init_node("python_orbbec_inference", anonymous=True)
    my_node = InferenceNodeRGBD(args)
    my_node.start()
    cv2.destroyAllWindows()
