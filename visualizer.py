#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import rospy
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose, Point
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from datetime import datetime
import time
import json
from utils import *

# remove numpy scientific notation
np.set_printoptions(suppress=True)

class VisualizerNode(object):
    def __init__(self, args):
        
        self.args = args
        
        self.rgb = None # Image frame
        self.depth = None # Image frame
        
        self.pcl_array_rgb = None
        self.pcl_array_xyz = None
        
        self.depth_cmap = get_mpl_colormap(args.depth_cmap)
        self.depth_array_max_threshold = 3000

        self.pcl_current_seq = -1        
        self.rgb_current_seq = -1
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
        self.save_dir_pcl_bin = os.path.join(self.save_dir, "pcl")

        if args.save or args.light_save:
            prInfo("Saving to {}/[rgb][depth][depth_color]".format(self.save_dir))
            if not os.path.exists(self.save_dir):
                prInfo("Creating directories to {}/[rgb][depth][depth_color]".format(self.save_dir))
                os.makedirs(self.save_dir)
                os.makedirs(self.save_dir_rgb)
                if not args.no_pcl:
                    os.makedirs(self.save_dir_pcl_bin)
                    
                if not args.no_depth and args.save:
                    os.makedirs(self.save_dir_depth)
                    os.makedirs(self.save_dir_depth_color)
                    
                args_dic = vars(args)
                with open(self.metadata, 'w') as fp:
                    json.dump(args_dic, fp)
                    
                prSuccess("Created directories to {}/[rgb][depth][depth_color][pcl]".format(self.save_dir))
                time.sleep(1)
            
 
        # Subscribers
        prInfo("Subscribing to {} for RGB".format(args.rgb_topic))
        self.rgb_sub = rospy.Subscriber(args.rgb_topic, Image, self.callback_rgb)
                
        if args.no_pcl:
            prWarning("No PCL subscriber because option --no_pcl is enabled")
        else:
            prInfo("Subscribing to {} for PCL".format(args.pcl_topic))
            self.pcl_sub = rospy.Subscriber(args.pcl_topic, PointCloud2, self.callback_pcl)
        
        if args.no_depth:
            prWarning("No depth subscriber because option --no_depth is enabled")
        else:
            prInfo("Subscribing to {} for depth".format(args.depth_topic))
            self.depth_sub = rospy.Subscriber(args.depth_topic,Image, self.callback_depth)

    def callback_pcl(self, msg):
        pcl_array = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width, -1))
        self.pcl_array_xyz = pcl_array[:,:,:3]
        self.pcl_array_rgb = pcl_array[:,:,3:]
        self.pcl_current_seq = msg.header.seq
        rospy.loginfo('pcl received ({})...'.format(msg.header.seq))

    def callback_rgb(self, msg):
        self.rgb = self.br.imgmsg_to_cv2(msg, "bgr8")
        self.rgb_current_seq = msg.header.seq
        rospy.loginfo('RGB received ({})...'.format(msg.header.seq))

    def callback_depth(self, msg):
        self.depth = self.br.imgmsg_to_cv2(msg, "mono16")
        self.depth_current_seq = msg.header.seq
        rospy.loginfo('Depth received ({})...'.format(msg.header.seq))

    def is_ready(self):
        ready = (self.rgb is not None) and (self.args.no_depth or self.depth is not None) and (self.args.no_pcl or self.pcl_array_xyz is not None)
        return ready

    def start(self):
        
        if self.args.light_save:
            # create dict for saving afterwards and avoid losing time
            saving_pcl = {}
            saving_rgb = {}
            
        while not rospy.is_shutdown():
                       
            if self.is_ready():
                
                image_count = self.current_image_count
                image_seq_unique = self.rgb_current_seq
                now = datetime.now()
                timestamp = now.strftime("%Y_%m_%d_%H_%M_%S_%f")
        
                if self.args.save or self.args.light_save:
                    rgb_path = os.path.join(self.save_dir_rgb, "{:08d}_seq_{:010d}_ts_{}.png".format(image_count, image_seq_unique, timestamp))
                    if self.args.save:
                        cv2.imwrite(rgb_path,  self.rgb)
                        prSuccess("Saved RGB to {}".format(rgb_path))
                    else:
                        saving_rgb[rgb_path] = self.rgb
                        
                rgb_array = np.asarray(self.rgb)

                if not self.args.no_show:
                    full_display_height = rgb_array.shape[0] if self.args.no_depth else rgb_array.shape[0] * 2
                    full_display_width = rgb_array.shape[1] if self.args.no_pcl else rgb_array.shape[1] * 2
                    full_display_array = np.zeros((full_display_height, full_display_width, 3), dtype = np.uint8)
                        
                    full_display_array[:rgb_array.shape[0], :rgb_array.shape[1] ,:] = rgb_array

                if self.args.no_depth:
                    depth_array = None
                else:
                    
                    if self.args.save:
                        depth_path = os.path.join(self.save_dir_depth, "{:08d}_seq_{:010d}_ts_{}.png".format(image_count, image_seq_unique, timestamp))
                        cv2.imwrite(depth_path,  self.depth)
                        prSuccess("Saved depth to {}".format(depth_path))   
                    
                    depth_array = np.asarray(self.depth)
                    depth_array[depth_array > self.depth_array_max_threshold] = self.depth_array_max_threshold

                    depth_array_disp = depth_array.copy()
                    depth_array_disp[depth_array_disp > 3000] = 3000              
                    depth_array_norm = ((depth_array_disp - depth_array_disp.min())) / (depth_array_disp.max() - depth_array_disp.min())
                    # depth_array_norm = ((depth_array - depth_array.min())) / (depth_array.max() - depth_array.min())
                    depth_array_norm = depth_array_norm * 255
                    depth_array_norm = depth_array_norm.astype(np.uint8)
                    depth_array_norm_colored = (self.depth_cmap[depth_array_norm] * 255).astype(np.uint8)

                    if self.args.save:
                        depth_color_path = os.path.join(self.save_dir_depth_color, "{:08d}_seq_{:010d}_ts_{}.png".format(image_count, image_seq_unique, timestamp))
                        cv2.imwrite(depth_color_path,  depth_array_norm_colored)
                        prSuccess("Saved depth color (scaled) to {}".format(depth_color_path))   
                    
                    if not self.args.no_show:
                        full_display_array[rgb_array.shape[0]:, :rgb_array.shape[1] ,:] = depth_array_norm_colored
                
                if self.args.no_pcl:
                    pcl_rgb_norm = None
                    pcl_xyz_norm = None
                else:
                    if self.args.save or self.args.light_save:
                        pcl_path = os.path.join(self.save_dir_pcl_bin, "{:08d}_seq_{:010d}_ts_{}.bin".format(image_count, image_seq_unique, timestamp))
                        
                        if self.args.save:
                            self.pcl_array_xyz.tofile(pcl_path)
                            prSuccess("Saved pcl to {}".format(pcl_path))   
                        elif self.args.light_save:
                            saving_pcl[pcl_path] = self.pcl_array_xyz
                            
                    if not self.args.no_show:
                        pcl_rgb_color = (self.pcl_array_rgb * 255).astype(np.uint8)
                        max_dist = 3.0 # 3m in any dimension
                        min_dist = -3.0 # 3m in any dimension
                        pcl_xyz_crop = self.pcl_array_xyz.copy()
                        pcl_xyz_crop[pcl_xyz_crop > max_dist] = max_dist
                        pcl_xyz_crop[pcl_xyz_crop < min_dist] = min_dist
                        pcl_dist_norm = (pcl_xyz_crop - min_dist) / (max_dist - min_dist)
                        pcl_dist_color = (pcl_dist_norm * 255).astype(np.uint8)
                        full_display_array[rgb_array.shape[0]:, rgb_array.shape[1]: ,:] = pcl_rgb_color[:,:,::-1]
                        full_display_array[:rgb_array.shape[0], rgb_array.shape[1]: ,:] = pcl_dist_color

                if not self.args.no_show:
                    #format(self.rgb_current_seq, self.depth_current_seq, self.pcl_current_seq)
                    cv2.imshow("RGBD window", full_display_array)
                    cv2.waitKey(3)
                    
                self.current_image_count += 1
                
                if (self.current_image_count > 1000 and self.args.light_save):
                    prWarning("Finish here and save all 100 images !")
                    self.rgb_sub.unregister()
                    if not self.args.no_pcl:
                        self.pcl_sub.unregister()
                    if not self.args.no_depth:
                        self.depth_sub.unregister()
                    break
                elif self.args.light_save:
                    prInfo("Collected image {} / 1000 before closing".format(self.current_image_count))            
                
            else:
                rospy.logwarn("Do not display/save images because not initialized (rgb or depth or pcl)")
           
            self.loop_rate.sleep()
        
        if self.args.light_save:
            
            prWarning("Please wait while we save images and pcl !")
            
            if not self.args.no_pcl:
                for key, value in saving_pcl.items():
                    value.tofile(key)
                    prSuccess("Saved pcl to {}".format(key)) 
                      
            for key, value in saving_rgb.items():
                cv2.imwrite(key, value)
                prSuccess("Saved rgb to {}".format(key))   
                
if __name__ == '__main__':

    ## Parser with params
    parser = ArgumentParser()
    parser.add_argument('--rgb_topic', default = "orbbec/rgb", type=str, help='ROS topic for RGB image')
    parser.add_argument('--depth_topic', default = "orbbec/depth", type=str, help='ROS topic for depth image')
    parser.add_argument('--pcl_topic', default = "orbbec/pcl", type=str, help='ROS topic for pcl')
    parser.add_argument(
        '--no_depth',
        action='store_true',
        default=False,
        help='Do not use depth subscriber / recorder / visualizer')
    parser.add_argument(
        '--no_pcl',
        action='store_true',
        default=False,
        help='Do not use pcl subscriber / recorder / visualizer')
    parser.add_argument(
        '--no_show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--save',
        action='store_true',
        default=False,
        help='whether to save images (rgb and d and pcl)')
    parser.add_argument(
        '--light_save',
        action='store_true',
        default=False,
        help='whether to save only rgb and pcl')
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Node and recording fps')
    parser.add_argument('--depth_cmap', default = "jet", type=str, help='mpl colormap for depth image')

    args = parser.parse_args()
    prInfo("Loaded with args : {}".format(args))
    
    rospy.init_node("python_orbbec_vis_save", anonymous=True)
    my_node = VisualizerNode(args)
    my_node.start()
    cv2.destroyAllWindows()