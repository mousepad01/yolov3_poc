import numpy as np
import tensorflow as tf
import cv2 as cv
import json

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
            size D x A x 2 (D = SCALE_COUNT)
        '''

        self.bool_anchor_masks = [[] for _ in range(SCALE_CNT)]
        self.target_anchor_masks = [[] for _ in range(SCALE_CNT)]
        '''
            for each scale,
                B x S[scale] x S[scale] x A x 1     - whether that anchor is responsible for an object or not
                B x S[scale] x S[scale] x A x 5     - regression targets and the class given by its one_hot index (but NOT one_hot encoded)            
        '''

        self.data_load_batch_size = data_load_batch_size

        self.img_size = img_size

    def load_images(self, purpose):
        '''
            generator, for lazy loading
            purpose: "train" | "validation"
        '''

        def _resize_with_pad(img):
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

            return cv.resize(img, self.img_size)
        
        if self.used_categories == {}:
            print("info not yet loaded")
            quit()

        current_loaded = []
        for img_id in self.imgs[purpose].keys():

            TESTIDS.append(img_id)

            img = cv.imread(self.data_path[purpose] + self.imgs[purpose][img_id]["filename"])
            img = _resize_with_pad(img)

            current_loaded.append(img)

            if len(current_loaded) == self.data_load_batch_size:

                current_loaded = tf.convert_to_tensor(current_loaded)
                yield current_loaded

                current_loaded = []

        if len(current_loaded) > 0:

            current_loaded = tf.convert_to_tensor(current_loaded)
            yield current_loaded

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

            off = (0, 0)
            ratio = 1

            if w < h:

                q1 = (h - w) // 2

                off = (q1, 0)
                ratio = self.img_size[0] / h

            elif h < w:

                q1 = (w - h) // 2

                off = (0, q1)
                ratio = self.img_size[0] / w

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

    # FIXME
    def determine_anchors(self):

        if self.used_categories == {}:
            print("info not yet loaded")
            quit()

        anchor_finder = AnchorFinder(self.imgs)
        self.anchors = anchor_finder.get_anchors()

    def assign_anchors_to_objects(self):

        # TODO
        def _iou(anchor, x, y, w, h):
            pass
        
        for img_id in self.imgs["train"].keys():

            bool_mask = [tf.zeros((GRID_CELL_CNT[d], GRID_CELL_CNT[d], ANCHOR_PERSCALE_CNT, 1), dtype=tf.int32) for d in range(SCALE_CNT)]
            target_mask = [tf.zeros((GRID_CELL_CNT[d], GRID_CELL_CNT[d], ANCHOR_PERSCALE_CNT, 5), dtype=tf.int32) for d in range(SCALE_CNT)]
            
            for bbox_d in self.imgs["train"][img_id]["objs"]:

                categ = np.int32(bbox_d["category"])
                x, y, w, h = bbox_d["bbox"]

                max_iou = -1
                max_iou_idx = None
                max_iou_scale = None

                for d in range(SCALE_CNT):
                    for a in range(ANCHOR_PERSCALE_CNT):

                        current_iou = _iou(self.anchors[d][a], x, y, w, h)
                        if current_iou > max_iou:
                            
                            max_iou = current_iou
                            max_iou_idx = a
                            max_iou_scale = d

                if max_iou > 0:

                    x, y = x + w // 2, y + h // 2
                    x, y = x / IMG_SIZE[0], y / IMG_SIZE[0]
                    x, y = x * GRID_CELL_CNT[max_iou_scale], y * GRID_CELL_CNT[max_iou_scale]

                    cx, cy = np.int32(np.floor(x)), np.int32(np.floor(y))
                    x, y = x - cx, y - cy

                    # get anchor w and h relative to the grid cell count
                    anchor_w = np.int32(np.floor(GRID_CELL_CNT[max_iou_scale] * (self.anchors[max_iou_scale][max_iou_idx][0] / IMG_SIZE[0])))
                    anchor_h = np.int32(np.floor(GRID_CELL_CNT[max_iou_scale] * (self.anchors[max_iou_scale][max_iou_idx][1] / IMG_SIZE[0])))

                    bool_mask[max_iou_scale][cx][cy][max_iou_idx] = tf.convert_to_tensor([1])
                    target_mask[max_iou_scale][cx][cy][max_iou_idx] = tf.convert_to_tensor([tf.sigmoid(x), 
                                                                                            tf.sigmoid(y),
                                                                                            tf.math.log(w / anchor_w), 
                                                                                            tf.math.log(h / anchor_h),
                                                                                            categ])

            for d in range(SCALE_CNT):

                self.bool_anchor_masks[d].append(bool_mask[d])
                self.target_anchor_masks[d].append(target_mask[d])

        for d in range(SCALE_CNT):

            self.bool_anchor_masks[d] = tf.convert_to_tensor(self.bool_anchor_masks[d])
            self.target_anchor_masks[d] = tf.convert_to_tensor(self.target_anchor_masks[d])
