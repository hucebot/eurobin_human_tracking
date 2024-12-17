#!/usr/bin/env python
# -*- coding: utf-8 -*-

# mmdet and mmpose import
from mmpose.apis import (get_track_id, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results,
                         vis_pose_tracking_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

# ros related import
import rospy
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose, Point
from cv_bridge import CvBridge

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

# utils import
from utils import *

# motion bert import
import imageio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_wild import WildDetDataset
from lib.utils.vismo import render_and_save

import copy

# remove numpy scientific notation
np.set_printoptions(suppress=True)


prSuccess("Everything imported !")


def crop_scale(motion, scale_range=[1, 1]):
    '''
        For input of MotionBERT
        Motion: [(M), T, 17, 3].
        Normalize to [-1, 1]
    '''
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2]!=0][:,:2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape)
    xmin = min(valid_coords[:,0])
    xmax = max(valid_coords[:,0])
    ymin = min(valid_coords[:,1])
    ymax = max(valid_coords[:,1])
    ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    scale = max(xmax-xmin, ymax-ymin) * ratio
    if scale==0:
        return np.zeros(motion.shape)
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    result[...,:2] = (motion[..., :2]- [xs,ys]) / scale
    result[...,:2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    return result


def coco2h36m(x):
    '''
        Input: x ((M )x T x V x C)
        
        COCO: {0-nose 1-Leye 2-Reye 3-Lear 4Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}
        
        H36M:
        0: 'root',
        1: 'rhip',
        2: 'rkne',
        3: 'rank',
        4: 'lhip',
        5: 'lkne',
        6: 'lank',
        7: 'belly',
        8: 'neck',
        9: 'nose',
        10: 'head',
        11: 'lsho',
        12: 'lelb',
        13: 'lwri',
        14: 'rsho',
        15: 'relb',
        16: 'rwri'
    '''
    y = np.zeros(x.shape)
    y[:,0,:] = (x[:,11,:] + x[:,12,:]) * 0.5
    y[:,1,:] = x[:,12,:]
    y[:,2,:] = x[:,14,:]
    y[:,3,:] = x[:,16,:]
    y[:,4,:] = x[:,11,:]
    y[:,5,:] = x[:,13,:]
    y[:,6,:] = x[:,15,:]
    y[:,8,:] = (x[:,5,:] + x[:,6,:]) * 0.5
    y[:,7,:] = (y[:,0,:] + y[:,8,:]) * 0.5
    y[:,9,:] = x[:,0,:]
    y[:,10,:] = (x[:,1,:] + x[:,2,:]) * 0.5
    y[:,11,:] = x[:,5,:]
    y[:,12,:] = x[:,7,:]
    y[:,13,:] = x[:,9,:]
    y[:,14,:] = x[:,6,:]
    y[:,15,:] = x[:,8,:]
    y[:,16,:] = x[:,10,:]
    return y


class InferenceNodeRGBD(object):
    def __init__(self, args):
        
        # init args
        self.args = args
        
        # init detector and pose
        prInfo('Initialiazing detector {}'.format(args.mb_checkpoint))        
        self.det_model = init_detector(
            args.det_config, args.det_checkpoint, device=args.device.lower())

        prInfo('Initialiazing 2D Pose model {}'.format(args.mb_checkpoint))        
        self.pose_model = init_pose_model(
            args.pose_config, args.pose_checkpoint, device=args.device.lower())

        # init 3d MotionBERT model
        prInfo('Initialiazing 3D Pose Lifter {}'.format(args.mb_checkpoint))        
        mb_3d_args = get_config(args.mb_3d_config)
        self.motionbert_3d_model = load_backbone(mb_3d_args)
        if torch.cuda.is_available():
            self.motionbert_3d_model = nn.DataParallel(self.motionbert_3d_model)
            self.motionbert_3d_model = self.motionbert_3d_model.cuda()
        else:
            prWarning("Expect cuda to be available but is_available returned false")
            exit(0)

        prInfo('Loading checkpoint {}'.format(args.mb_checkpoint))
        mb_checkpoint = torch.load(args.mb_checkpoint, map_location=lambda storage, loc: storage)
        self.motionbert_3d_model.load_state_dict(mb_checkpoint['model_pos'], strict=True)
        self.motionbert_3d_model.eval()
        prInfo('Loaded motionbert_3d_model')
        # no need for the whole WildDetDataset stuff, just manually make the input trajectories for the tracks

        # dataset params for detector and pose
        self.dataset = self.pose_model.cfg.data['test']['type']
        self.dataset_info = self.pose_model.cfg.data['test'].get('self.dataset_info', None)
        if self.dataset_info is None:
            warnings.warn(
                'Please set `self.dataset_info` in the config.'
                'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                DeprecationWarning)
        else:
            self.dataset_info = DatasetInfo(self.dataset_info)
        
        self.return_heatmap = False

        self.next_id = 0
        self.pose_results = []
        self.count_frames = 0
        self.tracks_in_current_image = {}

        ## Init for node and save path
        
        self.rgb = None # Image frame
        self.depth = None # Image frame

        self.pcl_array_rgb = None
        self.pcl_array_xyz = None
        
        self.depth_array_max_threshold = 20000 #3000 # does not apply when saving depth mono16 image
        
        # viewing options
        self.depth_cmap = get_mpl_colormap(args.depth_cmap)
        self.confidence_cmap = get_mpl_colormap("viridis")
        self.vis_img = None # output image RGB + detections
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

        # make the output path
        now = datetime.now()
        timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
        self.save_dir = os.path.join("output", "record_{:s}".format(timestamp))        
        self.metadata = os.path.join(self.save_dir, "metadata.json")
        self.save_dir_rgb = os.path.join(self.save_dir, "rgb")
        self.save_dir_depth = os.path.join(self.save_dir, "depth")
        self.save_dir_depth_color = os.path.join(self.save_dir, "depth_color")
        self.save_dir_result = os.path.join(self.save_dir, "output")
        self.save_dir_pcl_bin = os.path.join(self.save_dir, "pcl")

        if args.save or args.light_save:
            prInfo("Saving to {}/[rgb][depth][depth_color][output][pcl]".format(self.save_dir))
            if not os.path.exists(self.save_dir):
                prInfo("Creating directories to {}/[rgb][depth][depth_color][output][pcl]".format(self.save_dir))
                os.makedirs(self.save_dir)
                os.makedirs(self.save_dir_rgb)
                os.makedirs(self.save_dir_pcl_bin)

                if args.save:
                    os.makedirs(self.save_dir_depth)
                    os.makedirs(self.save_dir_depth_color)
                    os.makedirs(self.save_dir_result)

                args_dic = vars(args)
                with open(self.metadata, 'w') as fp:
                    json.dump(args_dic, fp)
                    
                prSuccess("Created directories to {}/[rgb][depth][depth_color][output][pcl]".format(self.save_dir))
                time.sleep(1)

        # Publishers
        self.goal_pub = rospy.Publisher('points/handover_goal', Point,  queue_size=10)

        # Subscribers
        prInfo("Subscribing to {} for RGB".format(args.rgb_topic))
        rospy.Subscriber(args.rgb_topic, Image,self.callback_rgb)
        prInfo("Subscribing to {} for depth".format(args.depth_topic))
        rospy.Subscriber(args.depth_topic,Image,self.callback_depth)
        prInfo("Subscribing to {} for PCL".format(args.pcl_topic))
        rospy.Subscriber(args.pcl_topic, PointCloud2, self.callback_pcl)


    def callback_pcl(self, msg):
        pcl_array = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width, -1))
        self.pcl_array_xyz = pcl_array[:,:,:3]
        self.pcl_array_rgb = pcl_array[:,:,3:]
        self.pcl_current_seq = msg.header.seq
        # rospy.loginfo('pcl received ({})...'.format(msg.header.seq))

    def callback_rgb(self, msg):
        self.rgb = self.br.imgmsg_to_cv2(msg, "bgr8")
        self.rgb_current_seq = msg.header.seq
        # rospy.loginfo('RGB received ({})...'.format(msg.header.seq))

    def callback_depth(self, msg):
        self.depth = self.br.imgmsg_to_cv2(msg, "mono16")
        self.depth_current_seq = msg.header.seq
        # rospy.loginfo('Depth received ({})...'.format(msg.header.seq))

    def is_ready(self):
        ready = (self.rgb is not None) and (self.depth is not None) and (self.pcl_array_xyz is not None)
        return ready
    
    def start(self):
        
        self.tracks = {} # all the tracks along time, we need to keep and history
        
        while not rospy.is_shutdown():
            
            if self.is_ready():
                
                image_count = self.current_image_count
                self.current_image_count += 1

                start_t = time.time()

                image_seq_unique = self.rgb_current_seq
                now = datetime.now()
                timestamp = now.strftime("%Y_%m_%d_%H_%M_%S_%f")
        
                if self.args.save or self.args.light_save:
                    rgb_path = os.path.join(self.save_dir_rgb, "{:08d}_seq_{:010d}_ts_{}.png".format(image_count, image_seq_unique, timestamp))
                    cv2.imwrite(rgb_path,  self.rgb)
                    prSuccess("Saved RGB to {}".format(rgb_path))
                    
                rgb_array = np.asarray(self.rgb)

                if self.args.save:
                    depth_path = os.path.join(self.save_dir_depth, "{:08d}_seq_{:010d}_ts_{}.png".format(image_count, image_seq_unique, timestamp))
                    cv2.imwrite(depth_path,  self.depth)
                    prSuccess("Saved depth to {}".format(depth_path))   
                
                depth_array = np.asarray(self.depth)
                depth_array[depth_array > self.depth_array_max_threshold] = self.depth_array_max_threshold
                
                assert(depth_array.shape[0] == rgb_array.shape[0])
                assert(depth_array.shape[1] == rgb_array.shape[1])
                
                # Process RGB array
                if self.last_inferred_seq < self.rgb_current_seq:
                    
                    prInfo("Do inference on frame {}".format(self.rgb_current_seq))

                    # keep old poses for tracking
                    pose_results_last = self.pose_results

                    tic = time.time()
                    mmdet_results = inference_detector(self.det_model, rgb_array) # list of detection rectangle i.e [(x1,y1,x2,y2), ...]
                    tac = time.time()
                    prInfo("Detection in {:.4f} sec (frame {}, number of human detection {})".format(tac-tic, self.rgb_current_seq, len(mmdet_results[0])))
                    
                    # keep the person class bounding boxes.
                    person_results = process_mmdet_results(mmdet_results, self.args.det_cat_id)   

                    tic = time.time()
                    # test a single image, with a list of bboxes.
                    self.pose_results, returned_outputs = inference_top_down_pose_model(
                        self.pose_model,
                        rgb_array,
                        person_results,
                        bbox_thr=self.args.bbox_thr,
                        format='xyxy',
                        dataset=self.dataset,
                        dataset_info=self.dataset_info,
                        return_heatmap=self.return_heatmap,
                        outputs=None)
                    tac = time.time()
                    prInfo("Poses in {:.4f} sec".format(tac-tic))

                    # get track id for each person instance
                    self.pose_results, self.next_id = get_track_id(
                        self.pose_results,
                        pose_results_last,
                        self.next_id,
                        use_oks=False,
                        tracking_thr=self.args.tracking_thr,
                        use_one_euro=self.args.euro,
                        fps=10)
                    
                    # produce an output image
                    if not self.args.no_show:
                        self.vis_img = rgb_array.copy()

                    if self.display_all_detection and not self.args.no_show:
                        for c in range(len(mmdet_results)):
                            if len(mmdet_results[c]) > 0:
                                for bi in range(mmdet_results[c].shape[0]):
                                    if mmdet_results[c][bi,4] > self.args.bbox_thr:
                                        bbox = mmdet_results[c][bi,:4].copy().astype(np.int32)
                                        bbox_ints = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                                        pt1 = ( min( max(0,bbox_ints[0]), depth_array.shape[1]),
                                                        min( max(0,bbox_ints[1]), depth_array.shape[0]) )
                                        pt2 = ( min( max(0,bbox_ints[2]), depth_array.shape[1]),
                                                        min( max(0,bbox_ints[3]), depth_array.shape[0]) )
                                        cv2.rectangle(self.vis_img, pt1, pt2, (255,255,255), 1)
                                        cv2.putText(self.vis_img, "{:s} ({:.0f}%)".format(YOLO_COCO_80_CLASSES[c], mmdet_results[c][bi,4]*100), pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    
                    #### post processing and 3D lifting ####
                                        
                    # remove too old tracks
                    for idx, track in list(self.tracks.items()):
                        if abs(self.current_image_count - track["last_seen"]) > self.args.max_frames_remove_tracks: 
                            prInfo("Removing track {}, not seen since frame {}, current is {}".format(idx, track["last_seen"], self.current_image_count))
                            self.tracks.pop(idx)
    
                    self.tracks_in_current_image =  {}
                    
                    for res in self.pose_results:
                        
                        # for each instance
                        
                        bbox = res["bbox"]
                        keypoints = res["keypoints"]
                        idx = res["track_id"]
                        
                        if idx not in self.tracks.keys():
                            prInfo("Adding a new track with idx {}".format(idx))
                            self.tracks[idx] = {}
                            self.tracks[idx]["last_seen"] = self.current_image_count
                            self.tracks[idx]["keypoints_2d"] = []
                        
                        # add keypoint to the current track
                        self.tracks[idx]["last_seen"] = self.current_image_count
                        self.tracks[idx]["keypoints_2d"].append(keypoints)
                        
                        self.tracks_in_current_image[idx] = {
                                                            "right_wrist_depth" : None,
                                                            "right_wrist_pose" : None,
                                                            "left_wrist_depth" : None,
                                                            "left_wrist_pose" : None,
                                                            "depth_center" : None,
                                                            "pose_center" : None,
                                                            "pose_from" : None
                                                            }
                        
                        # if history is long enough, process the trajectory for MotionBERT
                        if len(self.tracks[idx]["keypoints_2d"]) >= self.args.mb_clip_len:
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
                                prInfo("MotionBERT in {:.4f} sec".format(tac-tic))
                                                                
                                # root relative
                                predicted_3d_pos[:,:,0,:] = 0  # [1,T,17,3]
                                
                                # TODO : change it because a bit weird it is not aligned with 2D poses because of the history !
                                predicted_3d_pos_np = predicted_3d_pos[0,-1,:,:].cpu().numpy() # keep only the last prediction
                                if "keypoints_3d" in self.tracks[idx].keys():
                                    self.tracks[idx]["keypoints_3d"].append(predicted_3d_pos_np)
                                else:
                                    self.tracks[idx]["keypoints_3d"] = [predicted_3d_pos_np]
                            
                        
                        # Draw bounding bbox
                        bbox = bbox.astype(np.int32)
                        bbox_ints = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                        pt1 = ( min( max(0,bbox_ints[0]), depth_array.shape[1]),
                                        min( max(0,bbox_ints[1]), depth_array.shape[0]) )
                        pt2 = ( min( max(0,bbox_ints[2]), depth_array.shape[1]),
                                        min( max(0,bbox_ints[3]), depth_array.shape[0]) )
                        color = RANDOM_COLORS[idx % 255]
                        color_tuple = (int(color[0]), int(color[1]), int(color[2]))
                        
                        if not self.args.no_show:
                            cv2.rectangle( self.vis_img, pt1, pt2, color_tuple, 2)
                                                
                        body_center_joints = [] # to store center of lsho, rsho, lhip, rhip in pixels
                                                                        
                        for j in range(keypoints.shape[0]):
                            
                            kp = keypoints[j,:]
                            confidence = int(kp[2] * 255)
                            confidence_color = (self.confidence_cmap[min(255,confidence)]*255).astype(np.uint8) 

                            if confidence > self.args.kpt_thr and kp[0] > 0 and kp[1] > 0 and kp[0] < depth_array.shape[1] and kp[1] < depth_array.shape[0]:
                                
                                if (j == 5) or (j == 6) or (j == 11) or (j == 12):
                                    # one keypoint of the torso
                                    body_center_joints.append(kp)

                                if not self.args.no_show:
                                    # kp_color_tuple = (int(confidence_color[0]), int(confidence_color[1]), int(confidence_color[2]))
                                    cv2.circle(self.vis_img, (int(kp[0]), int(kp[1])), 2, color_tuple, thickness = 3)
                                
                                # if wrists, find depth and pose
                                
                                if (j == 10):
                                    # right wrist
                                    depth_wrist = depth_array[int(kp[1]), int(kp[0])]
                                    pose_wrist = self.pcl_array_xyz[int(kp[1]), int(kp[0]),:]
                                    self.tracks_in_current_image[idx]["right_wrist_depth"] = depth_wrist
                                    self.tracks_in_current_image[idx]["right_wrist_pose"] = pose_wrist
                                    if not self.light_display and not self.args.no_show:
                                        cv2.drawMarker(self.vis_img, (int(kp[0]), int(kp[1])), color = color_tuple, thickness = 3, 
                                                        markerType = cv2.MARKER_CROSS, line_type = cv2.LINE_AA,
                                                        markerSize = 16)
                                        cv2.putText(self.vis_img, "{:.0f}cm | {:.2f} {:.2f} {:.2f}".format(depth_wrist/10, pose_wrist[0], pose_wrist[1], pose_wrist[2]), (int(kp[0]), int(kp[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                                
                                elif (j == 9):
                                    # left wrist
                                    depth_wrist = depth_array[int(kp[1]), int(kp[0])]
                                    pose_wrist = self.pcl_array_xyz[int(kp[1]), int(kp[0]),:]
                                    self.tracks_in_current_image[idx]["left_wrist_depth"] = depth_wrist
                                    self.tracks_in_current_image[idx]["left_wrist_pose"] = pose_wrist
                                    if not self.light_display and not self.args.no_show:
                                        cv2.drawMarker(self.vis_img, (int(kp[0]), int(kp[1])), color = color_tuple, thickness = 3, 
                                                        markerType = cv2.MARKER_CROSS, line_type = cv2.LINE_AA,
                                                        markerSize = 16)
                                        cv2.putText(self.vis_img, "{:.0f}cm | {:.2f} {:.2f} {:.2f}".format(depth_wrist/10, pose_wrist[0], pose_wrist[1], pose_wrist[2]), (int(kp[0]), int(kp[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

                        # find the body center
                        if len(body_center_joints) == 4:
                            # if we managed to find the 4 points of the torso, search on the torso
                            body_center_joints = np.asarray(body_center_joints) # lsho, rsho, lhip, rhip
                            lsho = body_center_joints[0,:]
                            rsho = body_center_joints[1,:]
                            lhip = body_center_joints[2,:]
                            rhip = body_center_joints[3,:]
                            
                            # find 4 points between lsho and rhip and 4 points between rsho and lhip to find something more precise
                            seg_steps = [0.0, 0.25, 0.50, 0.75, 1.0]
                            depths_torso = []
                            poses_torso = []
                            for step in seg_steps:
                                
                                p1 = step * lsho + (1 - step) * rhip
                                if p1[0] < depth_array.shape[1] and p1[1] < depth_array.shape[0]:
                                    depth_p1 = depth_array[int(p1[1]), int(p1[0])]
                                    pose_p1 = self.pcl_array_xyz[int(p1[1]), int(p1[0]), :]
                                    if depth_p1 > 0:
                                        depths_torso.append(depth_p1)
                                        poses_torso.append(pose_p1)
                                    
                                p2 = step * rsho + (1 - step) * lhip
                                if p2[0] < depth_array.shape[1] and p2[1] < depth_array.shape[0]:
                                    depth_p2 = depth_array[int(p2[1]), int(p2[0])]
                                    pose_p2 = self.pcl_array_xyz[int(p2[1]), int(p2[0]), :]
                                    if depth_p2 > 0:
                                        depths_torso.append(depth_p2)
                                        poses_torso.append(pose_p2)
                                        
                                if not self.args.no_show:
                                    # draw to check
                                    cv2.drawMarker(self.vis_img, (int(p1[0]), int(p1[1])), color = color_tuple, thickness = 1, 
                                                    markerType = cv2.MARKER_DIAMOND, line_type = cv2.LINE_AA,
                                                    markerSize = 8)                            
                                    cv2.drawMarker(self.vis_img, (int(p2[0]), int(p2[1])), color = color_tuple, thickness = 1, 
                                                    markerType = cv2.MARKER_DIAMOND, line_type = cv2.LINE_AA,
                                                    markerSize = 8)  
                            
                            if len(depths_torso) > 3:
                                # at least 4 points to average decently
                                depth_body = np.asarray(depths_torso).mean()
                                pose_body = np.asarray(poses_torso).mean(axis = 0)
                                self.tracks_in_current_image[idx]["depth_center"] = depth_body # mm
                                self.tracks_in_current_image[idx]["pose_center"] = pose_body # m
                                self.tracks_in_current_image[idx]["pose_from"] = "torso"
                                        
                                # just for drawing
                                body_center = np.mean(body_center_joints, axis = 0)
                                # Draw center of body
                                body_center = (int(body_center[0]), int(body_center[1]))
                                if not self.light_display and not self.args.no_show:
                                    cv2.drawMarker(self.vis_img, body_center, color = color_tuple, thickness = 3, 
                                                    markerType = cv2.MARKER_TILTED_CROSS, line_type = cv2.LINE_AA,
                                                    markerSize = 16)
                                    cv2.putText(self.vis_img, "{:.0f}cm | {:.2f} {:.2f} {:.2f}".format(depth_body/10, pose_body[0], pose_body[1], pose_body[2]), (int(body_center[0]), int(body_center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
                                
                            # # fetch depth and pose at torso center
                            # if body_center[0] < depth_array.shape[1] and body_center[1] < depth_array.shape[0]:
                            #     depth_center = depth_array[body_center[1], body_center[0]]
                            #     pose_center = self.pcl_array_xyz[body_center[1], body_center[0],:]
                            #     if not self.light_display:
                            #         cv2.putText(self.vis_img, "{:.0f}cm | {:.2f} {:.2f} {:.2f}".format(depth_center/10, pose_center[0], pose_center[1], pose_center[2]), (int(body_center[0]), int(body_center[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                            #     if (depth_center != 0):
                            #         self.tracks_in_current_image[idx]["depth_center"] = depth_center # mm
                            #         self.tracks_in_current_image[idx]["pose_center"] = pose_center # m
                            #         self.tracks_in_current_image[idx]["pose_from"] = "torso"
                            #         # prSuccess("Publishing coordinates {:.2f} {:.2f} {:.2f}".format(pose_center[0], pose_center[1], pose_center[2]))
                            #         # self.goal_pub.publish(Point(x = pose_center[0], y = pose_center[1], z = pose_center[2]))
                            
                        else:
                            # if we did not managed to find the 4 points of the torso, search in the bbox
                            prWarning("Can't use body center from shoulders and hips, use center of box for track {} || UPDATE : do nothing".format(idx))
                            
                            if False:
                                # Draw center of bbox
                                bbox_center = (int(pt1[0]/2 + pt2[0]/2), int(pt1[1]/2 + pt2[1]/2))
                                if not self.light_display:
                                    cv2.drawMarker(self.vis_img, bbox_center, color = color_tuple, thickness = 3, 
                                                    markerType = cv2.MARKER_CROSS, line_type = cv2.LINE_AA,
                                                    markerSize = 16)
                                
                                # fetch depth and pose at bbox center
                                if bbox_center[0] < depth_array.shape[1] and bbox_center[1] < depth_array.shape[0]:
                                    depth_center = depth_array[bbox_center[1], bbox_center[0]]
                                    pose_center = self.pcl_array_xyz[bbox_center[1], bbox_center[0],:]
                                    if not self.light_display:
                                        cv2.putText(self.vis_img, "{:.0f}cm | {:.2f} {:.2f} {:.2f}".format(depth_center/10, pose_center[0], pose_center[1], pose_center[2]), (int(bbox_center[0]), int(bbox_center[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                                    if (depth_center != 0):
                                        self.tracks_in_current_image[idx]["depth_center"] = depth_center # mm
                                        self.tracks_in_current_image[idx]["pose_center"] = pose_center # m
                                        self.tracks_in_current_image[idx]["pose_from"] = "bbox"
                                        # prSuccess("Publishing coordinates {:.2f} {:.2f} {:.2255f}".format(pose_center[0], pose_center[1], pose_center[2]))
                                        # self.goal_pub.publish(Point(x = pose_center[0], y = pose_center[1], z = pose_center[2]))
                                    
                        # draw skeleton
                        if not self.args.no_show:
                            for limb in COCO17_JOINTS_LIMBS:
                                start = keypoints[limb[0],:]
                                end = keypoints[limb[1],:]
                                start_point = (int(start[0]), int(start[1])) 
                                end_point = (int(end[0]), int(end[1]))
                                if (start[2] > self.args.kpt_thr) and (end[2] > self.args.kpt_thr):
                                    cv2.line(self.vis_img, start_point, end_point, color = color_tuple, thickness = 3)
                        
                    min_depth = 1e6 # mm
                    min_depth_idx = -1
                    for idx, track_info in self.tracks_in_current_image.items():
                        depth = track_info["depth_center"]
                        if depth is not None:
                            if depth < min_depth:
                                min_depth = depth
                                min_depth_idx = idx
                    
                    if (min_depth_idx != -1):
                        pose_closest = self.tracks_in_current_image[min_depth_idx]["pose_center"]
                        prInfo("Using track {} as it is the closest".format(min_depth_idx))
                        self.goal_pub.publish(Point(x = pose_closest[0], y = pose_closest[1], z = pose_closest[2]))
                        prSuccess("Publishing coordinates {:.2f} {:.2f} {:.2f}".format(pose_closest[0], pose_closest[1], pose_closest[2]))
                        if not self.args.no_show:
                            cv2.putText(self.vis_img, "{:.2f} {:.2f} {:.2f}".format(pose_closest[0], pose_closest[1], pose_closest[2]), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 5)
                            cv2.putText(self.vis_img, "{:.2f} {:.2f} {:.2f}".format(pose_closest[0], pose_closest[1], pose_closest[2]), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
                    else:
                        if not self.args.no_show:
                            cv2.putText(self.vis_img, "No tracks with pose found", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 5)
                            cv2.putText(self.vis_img, "No tracks with pose found", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)           
                    
                    self.last_inferred_seq = self.rgb_current_seq

                    if self.args.save and not self.args.no_show:
                        results_path = os.path.join(self.save_dir_result, "{:08d}_seq_{:010d}_ts_{}.png".format(image_count, image_seq_unique, timestamp))
                        cv2.imwrite(results_path, self.vis_img)
                        prSuccess("Saved result to {}".format(results_path))   
                                    
                else:
                    prWarning("No inference because the current RGB frame has already been processed")
                
                if not self.args.no_show:
                    depth_array_norm = ((depth_array - depth_array.min())) / (depth_array.max() - depth_array.min())
                    depth_array_norm = depth_array_norm * 255
                    depth_array_norm = depth_array_norm.astype(np.uint8)
                    depth_array_norm_colored = (self.depth_cmap[depth_array_norm] * 255).astype(np.uint8)

                    if self.args.save:
                        depth_color_path = os.path.join(self.save_dir_depth_color, "{:08d}_seq_{:010d}_ts_{}.png".format(image_count, image_seq_unique, timestamp))
                        cv2.imwrite(depth_color_path,  depth_array_norm_colored)
                        prSuccess("Saved depth color (scaled) to {}".format(depth_color_path))   

                    if self.args.save or self.args.light_save:
                        pcl_path = os.path.join(self.save_dir_pcl_bin, "{:08d}_seq_{:010d}_ts_{}.bin".format(image_count, image_seq_unique, timestamp))
                        self.pcl_array_xyz.tofile(pcl_path)
                        prSuccess("Saved pcl to {}".format(pcl_path))   

                    if self.vis_img is not None:
                        full_display_array = np.zeros((rgb_array.shape[0] * 2, rgb_array.shape[1], 3), dtype = np.uint8)
                        full_display_array[:rgb_array.shape[0], : ,:] = self.vis_img 
                        full_display_array[rgb_array.shape[0]:, : ,:] = depth_array_norm_colored
                        
                        cv2.imshow("RGBD window", full_display_array)
                        cv2.waitKey(3)
                        
                end_t = time.time()
                prInfoBold("Processed frame {} in {:.4f} sec".format(self.current_image_count, end_t-start_t))

                    
                
                
            else:
                print("Images are None !")
                
            self.loop_rate.sleep()
    
if __name__ == '__main__':

    ## Parser with params
    parser = ArgumentParser()
    parser.add_argument('--det_config', type=str, default = "./configs/detection/yolov3_d53_320_273e_coco.py", help='Config file for detection')
    parser.add_argument('--det_checkpoint', type=str, default = "./models/yolov3_d53_320_273e_coco-421362b6.pth", help='Checkpoint file for detection')
    parser.add_argument('--pose_config', type=str, default = "./configs/pose/ViTPose_small_coco_256x192.py", help='Config file for pose')
    parser.add_argument('--pose_checkpoint', type=str, default = "./models/vitpose_small.pth", help='Checkpoint file for pose')
    parser.add_argument(
        '--device', 
        default='cuda:0',
        help='Device used for inference')
    parser.add_argument(
        '--det_cat_id',
        type=int,
        default=1,
        help='Category id for bounding box detection model (person)')
    parser.add_argument(
        '--bbox_thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt_thr', 
        type=float, 
        default=0.3, 
        help='Keypoint score threshold')
    parser.add_argument(
        '--tracking_thr', 
        type=float, 
        default=0.3, 
        help='Tracking threshold')
    parser.add_argument(
        '--euro',
        action='store_true',
        help='Using One_Euro_Filter for smoothing')

    parser.add_argument('--rgb_topic', default = "orbbec/rgb", type=str, help='ROS topic for RGB image')
    parser.add_argument('--depth_topic', default = "orbbec/depth", type=str, help='ROS topic for depth image')
    parser.add_argument('--pcl_topic', default = "orbbec/pcl", type=str, help='ROS topic for pcl')
    parser.add_argument(
        '--no_show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--save',
        action='store_true',
        default=False,
        help='whether to save images (rgb and d and predictions and pcl)')
    parser.add_argument(
        '--light_save',
        action='store_true',
        default=False,
        help='whether to save only rgb and pcl (not optimized use the light_save of visualizer for optimized saving)')
    parser.add_argument(
        '--display_all_detection', "-dad",
        action='store_true',
        default=False,
        help='whether to display all detections or only human')
    parser.add_argument(
        '--light_display', "-ld",
        action='store_true',
        default=False,
        help='whether to display only skeletons')
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Node and recording fps')
    parser.add_argument('--depth_cmap', default = "jet", type=str, help='mpl colormap for depth image')

    parser.add_argument('--mb_3d_config', type=str, default = "./configs/pose3d/MB_ft_h36m.yaml", help='Config file for 3D poses')
    parser.add_argument('--mb_checkpoint', type=str, default = "./checkpoint/pose3d/MB_train_h36m/best_epoch.bin", help='Checkpoint file for 3D poses')
    parser.add_argument(
        '--mb_clip_len',
        type=int,
        default=10,
        help='Number of past frames to use for MotionBERT (default in model is 243)')
    parser.add_argument(
        '--max_frames_remove_tracks',
        type=int,
        default=2,
        help='Number frames without the track present to keep going before removing a track')
    

    args = parser.parse_args()
    
    assert has_mmdet, 'Please install mmdet to run the demo.'
    assert args.det_config is not None
    assert args.det_checkpoint is not None
    
    if (args.save or args.light_save) and args.no_show:
        print("Do not use the no_show mode if save is enabled, no rendering is done if --no_show")
    
    prInfo("Loaded with args : {}".format(args))
    
    rospy.init_node("python_orbbec_inference", anonymous=True)
    my_node = InferenceNodeRGBD(args)
    my_node.start()
    cv2.destroyAllWindows()