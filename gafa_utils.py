

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize
import numpy as np
import torchvision.transforms.functional as TF
from torchvision import transforms
from albumentations.core.transforms_interface import DualTransform, to_tuple
import albumentations as A

import torchvision.transforms as T
from torchvision.transforms import Compose as ComposeTransform

import matplotlib.pyplot as plt

from PIL import Image

from utils import *

MIN_CONF_THRESH = 0.3
MIN_IDXS_COUNT = 50


class SingleAttrTransform:
    """
    Superclass for data transformation
    """

    def __init__(self, input_key, output_key):
        self.input_keys = self._validate_key_arg(input_key)
        self.output_keys = self._validate_key_arg(output_key)
        if len(self.input_keys) != len(self.output_keys):
            raise Exception(
                f"len(input_keys) != len(output_keys): {len(self.input_keys)} != {len(self.output_keys)}"
            )

    def __call__(self, item):
        """
        item: dictionary containing each variable in a dataset
        """
        self.before_transform(item)
        for in_key, out_key in zip(self.input_keys, self.output_keys):
            input_seq = item[in_key]
            item[out_key] = self.transform(input_seq)
        return item

    def transform(self, input_seq):
        raise NotImplementedError

    def before_transform(self, item):
        return

    def _validate_key_arg(self, key_or_keys):
        if isinstance(key_or_keys, str):
            return [key_or_keys]
        else:
            return key_or_keys


class ImageTransform:
    def __init__(self, img_key, transform):
        self.img_key = img_key
        self.transform = transform

    def __call__(self, item):
        item[self.img_key] = self.transform(item[self.img_key])
        return item

######################################
############ Bounding Box ##############
#####################################
class ExpandBB(SingleAttrTransform):
    """
    Expand or shurink the bounding box by multiplying specified arguments
    """

    def __init__(self, t, b, l, r, input_key="bb", output_key=None):
        output_key = output_key or input_key
        super().__init__(input_key, output_key)
        self.t = t
        self.b = b
        self.l = l
        self.r = r

    def transform(self, bb):
        old_w, old_h = bb["w"], bb["h"]
        old_u, old_v = bb["u"], bb["v"]

        lpad = int(old_w * self.l)
        rpad = int(old_w * self.r)
        tpad = int(old_h * self.t)
        bpad = int(old_h * self.b)

        return {
            "w": old_w + lpad + rpad,
            "h": old_h + tpad + bpad,
            "u": old_u - lpad,
            "v": old_v - tpad,
        }

class SquareFromWidth(SingleAttrTransform):
    """
    Expand or shurink the bounding box by multiplying specified arguments
    """

    def __init__(self, t, b, l, r, input_key="bb", output_key=None):
        output_key = output_key or input_key
        super().__init__(input_key, output_key)
        self.t = t
        self.b = b
        self.l = l
        self.r = r

    def transform(self, bb):
        old_w, old_h = bb["w"], bb["h"]
        old_u, old_v = bb["u"], bb["v"]

        lpad = 0 #int(old_w * self.l)
        rpad = 0 #int(old_w * self.r)
        tpad = 0 #int(old_h * self.t)
        bpad = 0 #int(old_h * self.b)

        return {
            "w": old_w + lpad + rpad,
            "h": old_h + tpad + bpad,
            "u": old_u - lpad,
            "v": old_v - tpad,
        }


class ExpandBBRect(SingleAttrTransform):
    """
    Make bonding box rectangle.
    """

    def __init__(self, input_key="bb", output_key=None):
        output_key = output_key or input_key
        super().__init__(input_key, output_key)

    def transform(self, bb):
        old_w, old_h = bb["w"], bb["h"]
        old_u, old_v = bb["u"], bb["v"]

        if old_w <= old_h:
            diff = old_h - old_w
            lpad = diff // 2

            return {"w": old_h, "h": old_h, "u": old_u - lpad, "v": old_v}

        if old_h < old_w:
            diff = old_w - old_h
            tpad = diff // 2

            return {"w": old_w, "h": old_w, "u": old_u, "v": old_v - tpad}


class ReshapeBBRect(SingleAttrTransform):
    """
    Crop or Expand the BB tp specified ratio
    """

    def __init__(self, img_ratio, input_key="bb", output_key=None):
        output_key = output_key or input_key
        super().__init__(input_key, output_key)

        assert len(img_ratio) == 2
        self.height = img_ratio[0]
        self.width = img_ratio[1]

    def transform(self, bb):
        old_w, old_h = bb["w"], bb["h"]
        old_u, old_v = bb["u"], bb["v"]

        old_ratio = old_h / old_w
        new_ratio = self.height / self.width

        # 縦が長すぎる場合
        if old_ratio > new_ratio:
            diff = old_h - old_w * (self.height / self.width)
            lpad = diff // 2

            return {"w": old_w, "h": old_h - diff, "u": old_u, "v": old_v + lpad}

        # 横が長すぎる場合
        else:
            diff = old_w - old_h * (self.width / self.height)
            lpad = diff // 2

            return {"w": old_w - diff, "h": old_h, "u": old_u + lpad, "v": old_v}


class CropBB:
    def __init__(self, img_key="image", bb_key="bb", out_key="image"):
        self.img_key = img_key
        self.bb_key = bb_key
        self.out_key = out_key

    def __call__(self, item):
        # self._check_keys(item)
        bb = item[self.bb_key]
        item[self.out_key] = TF.crop(
            item[self.img_key], top=int(bb["v"]), left=int(bb["u"]), height=int(bb["h"]), width=int(bb["w"])
        )
        return item


class KeypointsToBB:
    def __init__(self, kp_indices):
        if hasattr(kp_indices, "__iter__"):
            kp_indices = list(kp_indices)
        self.kp_indices = kp_indices

    def __call__(self, item):
        out = {k: v for k, v in item.items()}
        kp = item["keypoints"]

        kp = kp[self.kp_indices]
        kp = kp[np.all(kp != 0, axis=1), :]
        u, v = np.min(kp.astype(np.int64), axis=0)
        umax, vmax = np.max(kp.astype(np.int64), axis=0)
        out["bb"] = {"u": u, "v": v, "w": umax - u, "h": vmax - v}
        return out




# define transforms
head_transform = ComposeTransform(
    [
        # KeypointsToBB((0, 1, 15, 16, 17, 18)),
        KeypointsToBB((0,1,2,3,4,5,6)), #coco17 corresponding
        ExpandBB(0.85, -0.2, 0.1, 0.1, "bb"),
        ExpandBBRect("bb"),
    ]
)

# define transforms
head_transform_rest = ComposeTransform(
    [
        # KeypointsToBB((0, 1, 15, 16, 17, 18)),
        KeypointsToBB((0,1,2,3,4,5,6)), #coco17 corresponding
        ExpandBB(0.1, -0.2, 0.1, 0.1, "bb"),
        ExpandBBRect("bb"),
    ]
)

# define transforms
head_transform_face = ComposeTransform(
    [
        # KeypointsToBB((0, 1, 15, 16, 17, 18)),
        KeypointsToBB((0,1,2,3,4)), #coco17 corresponding
        ExpandBB(3.0, 2.5, 0.5, 0.5, "bb"),
        # ExpandBBRect("bb"),
    ]
)



body_transform = ComposeTransform(
    [
        KeypointsToBB(slice(None)),
        ExpandBB(0.15, 0.05, 0.2, 0.2, "bb"),
        ExpandBBRect("bb"),
        ReshapeBBRect((256, 192)),
        CropBB(bb_key="bb"),
        ImageTransform(
            "image",
            T.Compose(
                [
                    T.Resize((256, 192)),
                ]
            ),
        ),
    ]
)

body_transform_from_bb = ComposeTransform(
    [
        ExpandBB(0.15, 0.05, 0.2, 0.2, "bb"),
        ExpandBBRect("bb"),
        ReshapeBBRect((256, 192)),
        CropBB(bb_key="bb"),
        ImageTransform(
            "image",
            T.Compose(
                [
                    T.Resize((256, 192)),
                ]
            ),
        ),
    ]
)

normalize_img = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

normalize_img_torch = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

@timeit
def get_valid_ids(body_json):
    #  count valid detections per idx to find the valid ones
    idxs_count = {}
    for det in body_json:
        idx = det["idx"]
        kpts = np.array(det["keypoints"]).reshape((-1, 3))
        if (kpts[:, 2] > MIN_CONF_THRESH).all():
            if idx in idxs_count.keys():
                idxs_count[idx] += 1
            else:
                idxs_count[idx] = 1

    valid_idxs = []
    for idx, count in idxs_count.items():
        if count > MIN_IDXS_COUNT:
            valid_idxs.append(idx)

    return (valid_idxs)

@timeit
def get_valid_frames_by_keys(valid_idxs, body_results):
    out = {}
    for idx in valid_idxs:
        out[idx] = []

    for det in body_results:
        if det["idx"] in valid_idxs:
            kpts = np.array(det["keypoints"]).reshape((-1, 3))
            if (kpts[:, 2] > MIN_CONF_THRESH).all():

                # add the timestamp to the frame detection
                date_str = det["image_id"].split(".")[0].split("_ts_")[-1]
                date_format = '%Y_%m_%d_%H_%M_%S_%f'
                timestamp = datetime.strptime(date_str, date_format)
                det["timestamp"] = timestamp

                # check previous timestamp
                if len(out[det["idx"]]) > 0:
                    last_ts = out[det["idx"]][-1]["timestamp"]
                    diff_ts = (timestamp - last_ts).total_seconds()
                else:
                    diff_ts = 0

                assert(diff_ts >= 0)

                # if diff_ts < 0.3:
                #     # add the frame detection to the output dic by idx
                #     out[det["idx"]].append(det)
                # else:
                #     print(det["idx"], "Discard det because ts diff too high ({} > 0.2 s)".format(diff_ts), tag = "warning", tag_color = "yellow", color = "white")
                
                out[det["idx"]].append(det)

    return out



@timeit
def get_inputs(f_i, valid_frames, n_frames):
    
    if f_i < n_frames:
        # not enough past frames
        return None, None, None, None, None, None
    else:
        imgs = torch.zeros((1, n_frames, 3, 256, 192))
        head_masks = torch.zeros((1, n_frames, 1, 256, 192))
        body_dvs = torch.zeros((1, n_frames, 2))

        norm_body_center = np.zeros((n_frames, 2))


        sequences_ids = [f_i + off for off in range(-n_frames + 1, 1)]
        image_ids = []
        print(sequences_ids)
        for k, i in enumerate(sequences_ids):
            seq_frame_i = valid_frames[i]
            # load images
            image_ids.append(seq_frame_i["image_id"])
            image_path = os.path.join(images_root, seq_frame_i["image_id"])
            img_org = Image.open(image_path)
            kpts = np.array(seq_frame_i["keypoints"]).reshape((-1,3))
            assert((kpts[:,2] > MIN_CONF_THRESH).all())
            
            item = {
                "image": img_org,
                "keypoints": kpts[:, :2],
            }

            # get head bb in pixels
            head_trans = head_transform(item)
            head_bb = head_trans['bb']
            head_bb = np.array([head_bb['u'], head_bb['v'], head_bb['w'], head_bb['h']]).astype(np.float32)
            
            # get body bb in pixels
            body_trans = body_transform(item) 
            body_bb = body_trans['bb']
            body_bb = np.array([body_bb['u'], body_bb['v'], body_bb['w'], body_bb['h']])
            body_image = np.array(body_trans['image'])
            
            # change head bb to relative to body bb
            head_bb_abs = head_bb.copy()
            
            head_bb[0] -= body_bb[0]
            head_bb[1] -= body_bb[1]
            
            head_bb[0] = head_bb[0] / body_bb[2]
            head_bb[1] = head_bb[1] / body_bb[3]
            head_bb[2] = head_bb[2] / body_bb[2]
            head_bb[3] = head_bb[3] / body_bb[3]
                    
            # store body center
            norm_body_center[k,:] = (body_bb[[0, 1]] + body_bb[[2, 3]] / 2) / body_bb[[2,3]]
            
            # normalize image
            img = normalize_img(image = body_image)['image']
            img = torch.from_numpy(img.transpose(2, 0, 1))
            
            assert(img.shape[0] == 3)
            assert(img.shape[1] == 256)
            assert(img.shape[2] == 192)
            
            # create mask of head bounding box
            head_mask = torch.zeros(1, img.shape[1], img.shape[2])
            head_bb_int = head_bb.copy()
            head_bb_int[[0, 2]] *= img.shape[2]
            head_bb_int[[1, 3]] *= img.shape[1]
            head_bb_int[2] += head_bb_int[0]
            head_bb_int[3] += head_bb_int[1]
            head_bb_int = head_bb_int.astype(np.int64)
            head_bb_int[head_bb_int < 0] = 0
            
            print(head_bb, color = "red")
            print(head_bb_int, color = "red")
            head_mask[:, head_bb_int[1]:head_bb_int[3], head_bb_int[0]:head_bb_int[2]] = 1

            # assign
            head_masks[0, k, :, :, :] = head_mask
            imgs[0, k, :, :, :] = img
        
        # compute dv
        body_dvs[0, :, :] = torch.from_numpy(norm_body_center - np.roll(norm_body_center, shift=1, axis=0))
        
        return imgs, head_masks, body_dvs, head_bb_abs, image_ids, body_bb