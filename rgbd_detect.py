#!/usr/bin/env python
# -*- coding: utf-8 -*-

# mmdet and mmpose import
from mmpose.apis import (
    get_track_id,
    inference_top_down_pose_model,
    init_pose_model,
    process_mmdet_results,
    vis_pose_tracking_result,
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

# utils import
from utils import *

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

        self.next_id = 0
        self.pose_results = []
        self.count_frames = 0
        self.tracks_in_current_image = {}

        ## Init for node and save path

        self.rgb = None  # Image frame
        self.depth = None  # Image frame

        self.pcl_array_rgb = None
        self.pcl_array_xyz = None

        self.depth_array_max_threshold = (
            20000  # 3000 # does not apply when saving depth mono16 image
        )

        # viewing options
        self.depth_cmap = get_mpl_colormap(args.depth_cmap)
        self.confidence_cmap = get_mpl_colormap("viridis")
        self.vis_img = None  # output image RGB + detections
        self.view_all_classes_dets = True
        self.display_all_detection = args.display_all_detection
        self.light_display = args.light_display

        self.pcl_current_seq = -1
        self.rgb_current_seq = -1
        self.last_inferred_seq = -1
        self.depth_current_seq = -1
        self.current_image_count = 0

        self.br = CvBridge()

        prInfo("Setting node rate to {} fps".format(args.fps))
        self.loop_rate = rospy.Rate(args.fps)

        # make the output path
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

        # Publishers
        self.goal_pub = rospy.Publisher(
            args.namespace + "/human", TransformStamped, queue_size=1
        )

        self.tf_br = tf2_ros.TransformBroadcaster()

        # Subscribers
        rgb_topic = args.namespace + "/rgb"
        depth_topic = args.namespace + "/depth"
        pcl_topic = args.namespace + "/pcl"
        prInfo("Subscribing to {} for RGB".format(rgb_topic))
        rospy.Subscriber(rgb_topic, Image, self.callback_rgb)
        prInfo("Subscribing to {} for depth".format(depth_topic))
        rospy.Subscriber(depth_topic, Image, self.callback_depth)
        prInfo("Subscribing to {} for PCL".format(pcl_topic))
        rospy.Subscriber(pcl_topic, PointCloud2, self.callback_pcl)

        self.rgb_frame_id = None

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
                            0.5,
                            (255, 255, 255),
                            1,
                        )
                        
    def plot_mmdet_person_bbox(self, idx, bbox, array_shape):
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
        color_tuple = (int(color[0]), int(color[1]), int(color[2]))

        cv2.rectangle(self.vis_img, pt1, pt2, color_tuple, 2)
                
    def process_keypoints(self, keypoints, depth_array, idx):
        body_center_joints = (
            []
        )  # to store center of lsho, rsho, lhip, rhip in pixels
        color = RANDOM_COLORS[idx]
        color_tuple = (int(color[0]), int(color[1]), int(color[2]))

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
                    if not self.light_display and not self.args.no_show:
                        cv2.drawMarker(
                            self.vis_img,
                            (int(kp[0]), int(kp[1])),
                            color=color_tuple,
                            thickness=3,
                            markerType=cv2.MARKER_CROSS,
                            line_type=cv2.LINE_AA,
                            markerSize=16,
                        )
                        cv2.putText(
                            self.vis_img,
                            "{:.0f}cm | {:.2f} {:.2f} {:.2f}".format(
                                depth_wrist / 10,
                                pose_wrist[0],
                                pose_wrist[1],
                                pose_wrist[2],
                            ),
                            (int(kp[0]), int(kp[1])),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 0, 255),
                            2,
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
                    if not self.light_display and not self.args.no_show:
                        cv2.drawMarker(
                            self.vis_img,
                            (int(kp[0]), int(kp[1])),
                            color=color_tuple,
                            thickness=3,
                            markerType=cv2.MARKER_CROSS,
                            line_type=cv2.LINE_AA,
                            markerSize=16,
                        )
                        cv2.putText(
                            self.vis_img,
                            "{:.0f}cm | {:.2f} {:.2f} {:.2f}".format(
                                depth_wrist / 10,
                                pose_wrist[0],
                                pose_wrist[1],
                                pose_wrist[2],
                            ),
                            (int(kp[0]), int(kp[1])),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 0, 255),
                            2,
                        )
                        
        return body_center_joints

    def get_depth_and_poses_of_torso(self, depth_array, lsho, rsho, lhip, rhip, idx):

        color = RANDOM_COLORS[idx]
        color_tuple = (int(color[0]), int(color[1]), int(color[2]))

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
    
    def plot_body_pose_data(self, body_center, depth_body, pose_body, idx):

        color = RANDOM_COLORS[idx]
        color_tuple = (int(color[0]), int(color[1]), int(color[2]))

        cv2.drawMarker(
            self.vis_img,
            body_center,
            color = color_tuple,
            thickness=3,
            markerType=cv2.MARKER_TILTED_CROSS,
            line_type=cv2.LINE_AA,
            markerSize=16,
        )
        cv2.putText(
            self.vis_img,
            "{:.0f}cm | {:.2f} {:.2f} {:.2f}".format(
                depth_body / 10,
                pose_body[0],
                pose_body[1],
                pose_body[2],
            ),
            (int(body_center[0]), int(body_center[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            3,
        )
    
    def plot_skeleton_2d(self, keypoints, idx):

        color = RANDOM_COLORS[idx]
        color_tuple = (int(color[0]), int(color[1]), int(color[2]))

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
                    thickness=3,
                )

    def plot_det_text_info(self, pose_closest):
        if pose_closest is not None:
            cv2.putText(
                self.vis_img,
                "{:.2f} {:.2f} {:.2f}".format(
                    pose_closest[0], pose_closest[1], pose_closest[2]
                ),
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
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
                1.2,
                (0, 0, 0),
                3,
            )
        else:
            cv2.putText(
                self.vis_img,
                "No tracks with pose found",
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                5,
            )
            cv2.putText(
                self.vis_img,
                "No tracks with pose found",
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 0),
                3,
            )
    
    def start(self):

        while not rospy.is_shutdown():

            if self.is_ready():

                image_count = self.current_image_count
                image_seq_unique = self.rgb_current_seq
                now = datetime.now()
                timestamp = now.strftime("%Y_%m_%d_%H_%M_%S_%f")

                if self.args.save or self.args.light_save:
                    self.save_rgb(image_count, image_seq_unique, timestamp)

                rgb_array = np.array(self.rgb)

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
                    prInfo(
                        "Detection in {:.4f} sec (frame {}, number of human detection {})".format(
                            tac - tic, current_frame_processing, len(mmdet_results[0])
                        )
                    )

                    # keep the person class bounding boxes.
                    person_results = process_mmdet_results(
                        mmdet_results, self.args.det_cat_id
                    )

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
                    prInfo("Poses in {:.4f} sec".format(tac - tic))

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

                    #### post processing ####

                    self.tracks_in_current_image = {}

                    for res in self.pose_results:

                        # for each instance

                        bbox = res["bbox"]
                        keypoints = res["keypoints"]
                        idx = res["track_id"] % 255

                        self.tracks_in_current_image[idx] = {
                            "right_wrist_depth": None,
                            "right_wrist_pose": None,
                            "left_wrist_depth": None,
                            "left_wrist_pose": None,
                            "depth_center": None,
                            "pose_center": None,
                            "pose_from": None,
                        }

                        # Draw bounding bbox
                        bbox = bbox.astype(np.int32)
                        
                        if not self.args.no_show:
                            self.plot_mmdet_person_bbox(idx, bbox, depth_array.shape)

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

                        if not self.args.no_show:
                            self.plot_det_text_info(pose_closest)

                    else:
                        
                        if not self.args.no_show:
                            self.plot_det_text_info(None)


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
        default="orbbec_head",
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
        default=True,
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

    prInfo("Loaded with args : {}".format(args))

    rospy.init_node("python_orbbec_inference", anonymous=True)
    my_node = InferenceNodeRGBD(args)
    my_node.start()
    cv2.destroyAllWindows()
