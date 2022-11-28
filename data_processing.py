import json
import time

import cv2 as cv
import numpy as np
import tensorflow as tf

from anchor_kmeans import *
from utils import *

class DataManager:
    '''
        NOTE:
            * adapted for object detection with single-label classification (for simplicity)
            * uses only a subset from the entire COCO dataset (currently, food)
    '''

    TRAIN_DATA_PATH = "./data/train2017/"
    VALIDATION_DATA_PATH = "./data/val2017/"

    TRAIN_INFO_PATH = "./data/annotations/instances_train2017.json"
    VALIDATION_INFO_PATH = "./data/annotations/instances_val2017.json"

    def __init__(self, train_data_path=TRAIN_DATA_PATH,
                        train_info_path=TRAIN_INFO_PATH,
                        validation_data_path=VALIDATION_DATA_PATH,
                        validation_info_path=VALIDATION_INFO_PATH,

                        data_load_batch_size=DATA_LOAD_BATCH_SIZE,
                        img_size=IMG_SIZE,
                    ):
        
        self.data_path = {
                            "train": train_data_path,
                            "validation": validation_data_path
                            }
        self.info_path = {
                            "train": train_info_path,
                            "validation": validation_info_path
                            }

        self.used_categories = {}
        '''
            self.used_categories[category dataset ID] = {
                                                            "name": name, 
                                                            "supercategory": supercateg (name / id ????),
                                                            "onehot": one hot index
                                                        }
        '''

        self.onehot_to_name = {}
        '''
            self.onehot_to_name[one hot idx] = name
        '''

        self.category_onehot_to_id = []
        '''
            category_onehot_to_id[one hot index] = category ID from dataset
        '''
        
        self.imgs = {
                        "train": {},
                        "validation": {}
                    }
        '''
            imgs[purpose][img ID] = {
                                        "objs": [
                                                    {
                                                        "category": one hot index,
                                                        "bbox": (x, y, w, h) absolute values
                                                    }
                                                ],
                                        "filename": filename
                                    }
        '''

        self.anchors = []
        '''
            D x A x 2 (D = SCALE_COUNT)
        '''

        self.bool_anchor_masks = [[] for _ in range(SCALE_CNT)]
        '''
            for each scale,
                B x S[scale] x S[scale] x A x 1     - whether that anchor is responsible for an object or not
        '''
        self.target_anchor_masks = [[] for _ in range(SCALE_CNT)]
        '''
            for each scale,
                B x S[scale] x S[scale] x A x 5     - regression targets and the class given by its one_hot index (but NOT one_hot encoded)            
        '''

        self.DATA_LOAD_BATCH_SIZE = data_load_batch_size

        self.IMG_SIZE = img_size

    def resize_with_pad(self, img):
        '''
            returns resized image with black symmetrical padding
        '''

        w, h = img.shape[0], img.shape[1]

        if w < h:

            q1 = (h - w) // 2
            q2 = h - w - q1

            padl = np.zeros((q1, h, 3), dtype=np.uint8)
            padr = np.zeros((q2, h, 3), dtype=np.uint8)
        
            img = np.concatenate([padl, img, padr], 0)

        elif h < w:

            q1 = (w - h) // 2
            q2 = w - h - q1

            padl = np.zeros((w, q1, 3), dtype=np.uint8)
            padr = np.zeros((w, q2, 3), dtype=np.uint8)

            img = np.concatenate([padl, img, padr], 1)

        return tf.convert_to_tensor(cv.resize(img, self.IMG_SIZE))

    def load_images(self, purpose):
        '''
            generator, for lazy loading
            purpose: "train" | "validation"
        '''
        
        if self.used_categories == {}:
            print("info not yet loaded")
            quit()

        current_loaded = []
        for img_id in self.imgs[purpose].keys():

            img = cv.imread(self.data_path[purpose] + self.imgs[purpose][img_id]["filename"])
            img = self.resize_with_pad(img)

            current_loaded.append(img)

            if len(current_loaded) == self.DATA_LOAD_BATCH_SIZE:

                current_loaded = tf.convert_to_tensor(current_loaded)
                yield current_loaded

                current_loaded = []

        if len(current_loaded) > 0:

            current_loaded = tf.convert_to_tensor(current_loaded)
            yield current_loaded

    # FIXME NORMALIZATION
    def load_train_data(self, batch_size):

        if DATA_LOAD_BATCH_SIZE % batch_size != 0:
            print("data load batch size not a multiple of train batch size")
            quit()

        slice_idx = 0
        for imgs in self.load_images("train"):

            bool_mask_size1 = self.bool_anchor_masks[0][slice_idx * DATA_LOAD_BATCH_SIZE: (slice_idx + 1) * DATA_LOAD_BATCH_SIZE]
            target_mask_size1 = self.target_anchor_masks[0][slice_idx * DATA_LOAD_BATCH_SIZE: (slice_idx + 1) * DATA_LOAD_BATCH_SIZE]

            bool_mask_size2 = self.bool_anchor_masks[1][slice_idx * DATA_LOAD_BATCH_SIZE: (slice_idx + 1) * DATA_LOAD_BATCH_SIZE]
            target_mask_size2 = self.target_anchor_masks[1][slice_idx * DATA_LOAD_BATCH_SIZE: (slice_idx + 1) * DATA_LOAD_BATCH_SIZE]

            bool_mask_size3 = self.bool_anchor_masks[2][slice_idx * DATA_LOAD_BATCH_SIZE: (slice_idx + 1) * DATA_LOAD_BATCH_SIZE]
            target_mask_size3 = self.target_anchor_masks[2][slice_idx * DATA_LOAD_BATCH_SIZE: (slice_idx + 1) * DATA_LOAD_BATCH_SIZE]
            
            yield tf.cast(imgs, tf.float32) / 255.0, bool_mask_size1, target_mask_size1, bool_mask_size2, target_mask_size2, bool_mask_size3, target_mask_size3
                    
            slice_idx += 1

    def load_train_data_serial(self):
        
        idx = 0
        for img_id in self.imgs["train"].keys():

            img = cv.imread(self.data_path["train"] + self.imgs["train"][img_id]["filename"])
            img = self.resize_with_pad(img)

            yield tf.cast(img, tf.float32) / 255.0, self.bool_anchor_masks[idx: idx + 1], self.target_anchor_masks[idx: idx + 1]

            idx += 1

    # TODO load validation data (images, real boxes) to calculate precision, recall ???

    def load_info(self):
        '''
            load everything at once
            * it adjustes the bounding boxes absolute coordinates and does not keep the original size
                (the image itself will be resized when loaded in load_images())
        '''

        def _find_resize_factors(w, h):
            '''
                returns:
                    * offset(s) to add to absolute coordinates
                    * ratio to multiply with absolute coordinates
            '''

            if w < h:

                q1 = (h - w) // 2

                off = (q1, 0)
                ratio = self.IMG_SIZE[0] / h

            elif h < w:

                q1 = (w - h) // 2

                off = (0, q1)
                ratio = self.IMG_SIZE[0] / w

            else:

                off = (0, 0)
                ratio = self.IMG_SIZE[0] / w    # or h

            return off, ratio

        for purpose in ["train", "validation"]:

            with open(self.info_path[purpose], "r") as info_f:

                info = info_f.read()
                info = json.loads(info)

            if self.used_categories == {}:
                for categ in info["categories"]:

                    if categ["supercategory"] != "food":
                        continue

                    self.used_categories[categ["id"]] = {
                                                            "name": categ["name"], 
                                                            "supercategory": categ["supercategory"],
                                                            "onehot": len(self.category_onehot_to_id)
                                                        }

                    self.onehot_to_name[len(self.category_onehot_to_id)] = categ["name"]
                    self.category_onehot_to_id.append(categ["id"])

            for anno in info["annotations"]:
                
                if anno["category_id"] not in self.used_categories:
                    continue
                
                if anno["image_id"] not in self.imgs[purpose].keys():
                    self.imgs[purpose][anno["image_id"]] = {
                                                                "objs": [],
                                                                "filename": ""
                                                            }

                # h, w intentionally reverted
                self.imgs[purpose][anno["image_id"]]["objs"].append({
                                                                        "category": self.used_categories[anno["category_id"]]["onehot"],
                                                                        "bbox": (anno["bbox"][1], anno["bbox"][0], anno["bbox"][3], anno["bbox"][2]) 
                                                                    })
            
            for img_info in info["images"]:

                if img_info["id"] not in self.imgs[purpose]:
                    continue
                
                self.imgs[purpose][img_info["id"]]["filename"] = img_info["file_name"]

                # h, w intentionally inverted
                h = img_info["width"]
                w = img_info["height"]

                off, ratio = _find_resize_factors(w, h)

                if off[0] > 0:

                    for bbox_d in self.imgs[purpose][img_info["id"]]["objs"]:
                        bbox_d["bbox"] = (off[0] + bbox_d["bbox"][0], bbox_d["bbox"][1], 
                                            bbox_d["bbox"][2], bbox_d["bbox"][3])

                elif off[1] > 0:

                    for bbox_d in self.imgs[purpose][img_info["id"]]["objs"]:
                        bbox_d["bbox"] = (bbox_d["bbox"][0], off[1] + bbox_d["bbox"][1], 
                                            bbox_d["bbox"][2], bbox_d["bbox"][3])
   
                for bbox_d in self.imgs[purpose][img_info["id"]]["objs"]:
                    bbox_d["bbox"] = (np.int32(np.floor(ratio * bbox_d["bbox"][0])), np.int32(np.floor(ratio * bbox_d["bbox"][1])), 
                                        np.int32(np.floor(ratio * bbox_d["bbox"][2])), np.int32(np.floor(ratio * bbox_d["bbox"][3])))

    def determine_anchors(self):

        if self.used_categories == {}:
            print("info not yet loaded")
            quit()

        anchor_finder = AnchorFinder(self.imgs)
        self.anchors = tf.cast(tf.convert_to_tensor(anchor_finder.get_anchors()), tf.int32)

    # FIXME
    def assign_anchors_to_objects(self):

        if self.used_categories == {}:
            print("info not yet loaded")
            quit()

        if self.anchors == []:
            print("anchors not yet determined")
            quit()

        def _iou(anchor, w, h):
            
            intersect_w = np.minimum(anchor[0], w)
            intersect_h = np.minimum(anchor[1], h)

            intersection = intersect_w * intersect_h
            union = w * h + anchor[0] * anchor[1] - intersection

            return intersection / union
        
        cnt_ = 0
        
        for img_id in self.imgs["train"].keys():

            cnt_ += 1
            if cnt_ > 64:
                break

            bool_mask = []
            target_mask = []

            for d in range(SCALE_CNT):

                bool_mask.append([[[[0] for _ in range(ANCHOR_PERSCALE_CNT)] for _ in range(GRID_CELL_CNT[d])] for _ in range(GRID_CELL_CNT[d])])
                target_mask.append([[[[0 for _ in range(4 + len(self.category_onehot_to_id))] for _ in range(ANCHOR_PERSCALE_CNT)] for _ in range(GRID_CELL_CNT[d])] for _ in range(GRID_CELL_CNT[d])])

            for bbox_d in self.imgs["train"][img_id]["objs"]:

                categ = np.int32(bbox_d["category"])
                x, y, w, h = bbox_d["bbox"]

                max_iou = -1
                max_iou_idx = None
                max_iou_scale = None

                for d in range(SCALE_CNT):
                    for a in range(ANCHOR_PERSCALE_CNT):

                        current_iou = _iou(self.anchors[d][a], w, h)
                        if current_iou > max_iou:
                            
                            max_iou = current_iou
                            max_iou_idx = a
                            max_iou_scale = d

                x, y = x + w // 2, y + h // 2
                x, y, w, h = x / IMG_SIZE[0], y / IMG_SIZE[0], w / IMG_SIZE[0], h / IMG_SIZE[0]
                x, y, w, h = x * GRID_CELL_CNT[max_iou_scale], y * GRID_CELL_CNT[max_iou_scale], w * GRID_CELL_CNT[max_iou_scale], h * GRID_CELL_CNT[max_iou_scale]

                cx, cy = np.int32(np.floor(x)), np.int32(np.floor(y))
                x, y = x - cx, y - cy   # x - cx, y - cy == sigmoid(tx) - cx, sigmoid(ty) - cy <=> x = sigmoid(tx), y = sigmoid(ty)

                # get anchor w and h relative to the grid cell count
                anchor_w = GRID_CELL_CNT[max_iou_scale] * (self.anchors[max_iou_scale][max_iou_idx][0] / IMG_SIZE[0])
                anchor_h = GRID_CELL_CNT[max_iou_scale] * (self.anchors[max_iou_scale][max_iou_idx][1] / IMG_SIZE[0])

                bool_mask[max_iou_scale][cx][cy][max_iou_idx] = [1]
                target_mask[max_iou_scale][cx][cy][max_iou_idx] = tf.concat([tf.convert_to_tensor([tf.math.log(x / (1 - x))]), 
                                                                            tf.convert_to_tensor([tf.math.log(y / (1 - y))]),
                                                                            tf.convert_to_tensor([tf.math.log(w / anchor_w)]), 
                                                                            tf.convert_to_tensor([tf.math.log(h / anchor_h)]),
                                                                            tf.cast(tf.one_hot(categ, len(self.category_onehot_to_id)), dtype=tf.double)],
                                                                            axis=0)

            for d in range(SCALE_CNT):

                self.bool_anchor_masks[d].append(bool_mask[d])
                self.target_anchor_masks[d].append(target_mask[d])

        for d in range(SCALE_CNT):

            self.bool_anchor_masks[d] = tf.convert_to_tensor(self.bool_anchor_masks[d], dtype=tf.float32)
            self.target_anchor_masks[d] = tf.convert_to_tensor(self.target_anchor_masks[d], dtype=tf.float32)
