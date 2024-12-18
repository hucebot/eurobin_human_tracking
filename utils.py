#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os

key = 'VERBOSE'
VERBOSITY = os.getenv(key) 
# 0 = only warnings, errors, success and debug, 1 = + info, 2 = + timer
if VERBOSITY is not None:
  VERBOSITY = int(VERBOSITY)
else:
  VERBOSITY = 2

print("Set verbosity level to : ", VERBOSITY)

import numpy as np
import matplotlib.pyplot as plt
from print_color import print
import copy

from functools import wraps
import time
import cv2

TEXT_SCALE = 1.0

def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 3).astype(np.float32) / 255.0


def prSuccess(text):
  print(text, tag = "ok", tag_color = "green", color = "white")

def prInfo(text):
  if VERBOSITY >= 1:
    print(text, tag = "info", tag_color = "cyan", color = "white")

def prTimer(text, tic, tac):
  if VERBOSITY >= 2:
    print("{} {:.0f} ms".format(text, (tac-tic)*1000), tag = "timer", tag_color = "purple", color = "white")

def prInfoBold(text):
  if VERBOSITY >= 1:
    print(text, tag = "info", tag_color = "cyan", color = "white", format = "bold")

def prDebug(text):
  print(text, tag = "debug", tag_color = "red", background = "white", color = "white")

def prWarning(text):
  print(text, tag = "warning", tag_color = "yellow", color = "white")

def prError(text):
  print(text, tag = "error", tag_color = "red", color = "white")


def draw_bbox_with_corners(image, bbox, color=(0, 255, 0), thickness=2, proportion=0.2):
    x_min, y_min, x_max, y_max = map(int, bbox[:4])

    width = x_max - x_min
    height = y_max - y_min

    corner_length = int(proportion * min(width, height))

    # Draw the rectangle
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 1)

    # Top-left corner
    cv2.line(image, (x_min, y_min), (x_min + corner_length, y_min), color, thickness)
    cv2.line(image, (x_min, y_min), (x_min, y_min + corner_length), color, thickness)

    # Top-right corner
    cv2.line(image, (x_max, y_min), (x_max - corner_length, y_min), color, thickness)
    cv2.line(image, (x_max, y_min), (x_max, y_min + corner_length), color, thickness)

    # Bottom-left corner
    cv2.line(image, (x_min, y_max), (x_min, y_max - corner_length), color, thickness)
    cv2.line(image, (x_min, y_max), (x_min + corner_length, y_max), color, thickness)

    # Bottom-right corner
    cv2.line(image, (x_max, y_max), (x_max, y_max - corner_length), color, thickness)
    cv2.line(image, (x_max, y_max), (x_max - corner_length, y_max), color, thickness)

YOLO_COCO_80_CLASSES = [
"person",
"bicycle",
"car",
"motorbike",
"aeroplane",
"bus",
"train",
"truck",
"boat",
"traffic light",
"fire hydrant",
"stop sign",
"parking meter",
"bench",
"bird",
"cat",
"dog",
"horse",
"sheep",
"cow",
"elephant",
"bear",
"zebra",
"giraffe",
"backpack",
"umbrella",
"handbag",
"tie",
"suitcase",
"frisbee",
"skis",
"snowboard",
"sports ball",
"kite",
"baseball bat",
"baseball glove",
"skateboard",
"surfboard",
"tennis racket",
"bottle",
"wine glass",
"cup",
"fork",
"knife",
"spoon",
"bowl",
"banana",
"apple",
"sandwich",
"orange",
"broccoli",
"carrot",
"hot dog",
"pizza",
"donut",
"cake",
"chair",
"sofa",
"pottedplant",
"bed",
"diningtable",
"toilet",
"tvmonitor",
"laptop",
"mouse",
"remote",
"keyboard",
"cell phone",
"microwave",
"oven",
"toaster",
"sink",
"refrigerator",
"book",
"clock",
"vase",
"scissors",
"teddy bear",
"hair drier",
"toothbrush"]

COCO17_JOINTS_LIMBS =       [[0,1],  [0,2],  [1,2],  [1,3],  [4,2],  [3,5],  [4,6],  [5,7],[6,8],[7,9],[8,10],      [5,6],[11,12],    [5,11],[6,12],[11,13],[12,14],[13,15],[14,16]]  

RANDOM_COLORS = np.array([
  [205, 150, 194],
  [ 17, 155, 211],
  [162, 121, 186],
  [194, 242,  27],
  [248,  79,  81],
  [134, 159, 164],
  [163,   7,  30],
  [ 93,   9, 121],
  [ 95,  54, 131],
  [ 77,  23,  22],
  [ 43,  17, 191],
  [ 34, 198, 162],
  [ 53,   5, 221],
  [ 37,  74,  55],
  [ 88, 204, 179],
  [200,  84, 192],
  [ 71,  75,  96],
  [  5, 250, 149],
  [  9, 216, 221],
  [ 54, 115,  69],
  [109,  92,  97],
  [186, 191, 222],
  [ 14,  41, 194],
  [ 75, 246, 175],
  [135, 112,  74],
  [ 18, 185,  33],
  [236, 129,  68],
  [ 58, 226, 186],
  [ 56,  63,  90],
  [231,  40, 251],
  [222, 112, 249],
  [ 77,  37, 189],
  [137,  94, 131],
  [170, 233,  53],
  [235,  29,  21],
  [ 66,  96,  46],
  [ 62,  29, 142],
  [ 12, 193,  90],
  [224, 151, 242],
  [132, 221, 176],
  [ 94,  75, 130],
  [157, 220, 166],
  [156,  47, 225],
  [ 76, 176, 108],
  [186, 189,  33],
  [139, 223,  78],
  [ 98, 169,  49],
  [ 39, 154,  71],
  [ 49, 191, 100],
  [128, 170,  25],
  [ 90, 127, 185],
  [180, 213, 170],
  [ 53, 153, 220],
  [109, 211,  12],
  [ 72, 125,  73],
  [126, 220, 193],
  [238,  38, 220],
  [ 77,  76,  46],
  [254, 186, 161],
  [126, 226, 187],
  [190, 142,  14],
  [132, 146, 254],
  [ 34,  39, 219],
  [ 78, 114, 127],
  [248, 145, 165],
  [145,  64,  10],
  [237,  84,  14],
  [ 18, 245, 229],
  [246,  40, 125],
  [187, 210,  10],
  [128, 197, 159],
  [152, 179, 221],
  [ 18, 159,  88],
  [ 17, 205, 133],
  [243, 111, 152],
  [ 86,  60, 202],
  [178,  71, 105],
  [ 49, 141, 244],
  [238, 169,  59],
  [ 91, 190,  81],
  [194, 113, 124],
  [209, 214, 138],
  [ 61, 251, 148],
  [113,  75, 124],
  [182, 147,   1],
  [ 86, 119, 160],
  [ 12, 253, 136],
  [149,  38,  41],
  [183, 161,  19],
  [153,   4,  68],
  [195, 147, 156],
  [165,  30, 189],
  [ 82,  55, 244],
  [ 33,  25, 248],
  [ 71, 193, 228],
  [244,  37, 174],
  [203,   6, 202],
  [118, 209, 136],
  [248, 144,  49],
  [  8, 145, 128],
  [164,  24,   0],
  [ 97, 196,  92],
  [243, 146, 179],
  [ 77, 144, 104],
  [134,  63,  50],
  [108, 155, 104],
  [200, 124, 251],
  [ 70,  35, 156],
  [115,  57, 148],
  [249, 236,   2],
  [119, 245,  43],
  [ 49, 101,  88],
  [ 27, 188,  88],
  [225,  20,  89],
  [ 94, 249, 118],
  [  1, 150,  65],
  [161,  77, 221],
  [144, 227, 134],
  [ 28, 231,  69],
  [165, 141, 223],
  [134, 124, 162],
  [151,  18, 210],
  [ 15,  39, 228],
  [ 88, 192,  62],
  [179,  36, 209],
  [ 99,  11, 191],
  [145,  76, 117],
  [183, 212, 247],
  [ 10,  52, 119],
  [154, 218, 200],
  [194, 227, 179],
  [  9,  73,   9],
  [ 66,  19,  65],
  [ 62, 201, 224],
  [ 18, 100, 101],
  [  4,  29, 246],
  [ 94,  47, 167],
  [ 57,  85, 162],
  [196, 245, 113],
  [234,  87, 229],
  [ 30, 199,  34],
  [ 41, 216, 200],
  [ 93, 155, 214],
  [236, 132,  87],
  [193, 191,  13],
  [222, 140, 102],
  [ 50, 194,  63],
  [244, 103,  90],
  [ 63, 234,  10],
  [ 45, 138, 147],
  [107,  11, 164],
  [ 93, 196,  79],
  [ 85,  20, 227],
  [  2,  74,   5],
  [155, 243,  68],
  [133, 102,  92],
  [ 85,  27, 104],
  [ 73,  69,  71],
  [176, 159, 175],
  [124, 113, 197],
  [102, 221,  40],
  [167, 164, 166],
  [214,   8,  43],
  [183, 139, 224],
  [130,  21,  83],
  [172,  11, 186],
  [199, 183, 201],
  [180, 166,  98],
  [ 28,  22, 177],
  [  4, 227,  64],
  [131,   2,  95],
  [  2, 164,  73],
  [ 89, 247,   7],
  [235,  93, 169],
  [ 51, 230,  61],
  [144, 144, 234],
  [157,  22,  89],
  [  0,  48, 113],
  [207,  63, 161],
  [200,   3, 166],
  [ 25,  92, 209],
  [243, 201, 247],
  [117,  78, 126],
  [229,  99, 105],
  [ 52, 184, 198],
  [ 29, 127, 174],
  [251, 113,  46],
  [220, 148,  28],
  [ 18, 228,  18],
  [216, 178,  17],
  [ 78,  54, 148],
  [223, 253, 150],
  [105,  69,  50],
  [229, 162,  35],
  [140,  47, 200],
  [103, 195, 216],
  [169,  23,  47],
  [ 73, 208,  20],
  [ 53, 184, 113],
  [225, 211,  40],
  [135, 163, 142],
  [243, 236,  67],
  [ 14,  20,  61],
  [ 11,  27, 107],
  [ 24, 145,  99],
  [155, 150, 243],
  [254, 153, 114],
  [ 91, 182, 222],
  [ 71, 216,  39],
  [  9,  55, 216],
  [144,   1, 144],
  [163, 166, 208],
  [149,  53,  64],
  [230,  45,  52],
  [171, 157,   2],
  [191,  43, 172],
  [180,  84, 131],
  [  8,  40,  88],
  [155,  63, 149],
  [196, 150, 149],
  [123, 219,  46],
  [  9,  63, 186],
  [ 19,  54, 155],
  [ 25,  43,  88],
  [140, 174, 131],
  [ 23, 158,  90],
  [152, 141, 207],
  [ 28, 160,  67],
  [ 17,  54, 220],
  [ 12, 186,   7],
  [129,  17,  94],
  [221,  84, 128],
  [142, 172, 202],
  [161, 214, 106],
  [ 75, 208, 229],
  [140,  39, 192],
  [183, 116, 110],
  [ 73, 104, 186],
  [152, 191, 227],
  [254,   1,  97],
  [193, 189,  73],
  [187, 108, 152],
  [ 86, 224,  29],
  [212, 192, 223],
  [130, 109,  55],
  [149, 130, 121],
  [ 70, 125,  16],
  [203,  54, 194],
  [ 23,  91, 249],
  [ 43,  73,   5],
  [  5, 165, 112],
  [189, 148, 214],
  [170,  56, 203],
  [ 69,  45,  90],
  [ 27, 169, 222],
  [187,  80,  33]
])



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
  
  
def timeit(func): 
    @wraps(func)
    def wrapper_function(*args, **kwargs): 
        tic = time.time()
        res = func(*args,  **kwargs) 
        tac = time.time()
        if VERBOSITY >= 2:
          print("{} {:.0f} ms".format(func.__name__, (tac-tic)*1000), tag = "timer", tag_color = "purple", color = "white")
        return res
    return wrapper_function 
