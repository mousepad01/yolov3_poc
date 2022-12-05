import json
import time
import pickle
import psutil

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

    CACHE_PATH = "./cache_entries/"
    FAKE_CACHE_KEY = "tmp_gt"

    def __init__(self, train_data_path=TRAIN_DATA_PATH,
                        train_info_path=TRAIN_INFO_PATH,
                        validation_data_path=VALIDATION_DATA_PATH,
                        validation_info_path=VALIDATION_INFO_PATH,

                        img_size=IMG_SIZE,
                        cache_key=None,
                    ):

        assert(cache_key != DataManager.FAKE_CACHE_KEY)
        
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
            * cacheable
        '''

        self.IMG_SIZE = img_size

        self.cache_key = cache_key
        '''
            determining anchors and assigning them to each bounding box can be done only once for a specific dataset
            * if cache_key is given and there is such cache, it upload data cached under this key when specific operations are called; 
            * if cache_key is given but there is no such cache, it will create it;
        '''

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
            loads image in batches
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

            if len(current_loaded) == DATA_LOAD_BATCH_SIZE:

                current_loaded = tf.convert_to_tensor(current_loaded)
                yield current_loaded

                current_loaded = []

        if len(current_loaded) > 0:

            current_loaded = tf.convert_to_tensor(current_loaded)
            yield current_loaded

    def load_gt(self, purpose):
        '''
            loads ground truth for each image batch
        '''

        if self.cache_key is None:
            cache_key = DataManager.FAKE_CACHE_KEY
        else:
            cache_key = self.cache_key

        IMG_CNT = len(self.imgs[purpose].keys())
        GT_BATCH_CNT = IMG_CNT // GT_LOAD_BATCH_SIZE

        incomplete = ((IMG_CNT % GT_LOAD_BATCH_SIZE) > 0)

        DATA_BATCH_PER_GT_BATCH = GT_LOAD_BATCH_SIZE // DATA_LOAD_BATCH_SIZE

        # declare outside loop in case total image count < GT_LOAD_BATCH_SIZE
        gt_batch_idx = 0

        for gt_batch_idx in range(GT_BATCH_CNT):

            print(f"gt b idx {gt_batch_idx}")

            with open(f"{self.CACHE_PATH}/{cache_key}_bool_masks_{purpose}_{gt_batch_idx}.bin", "rb") as cache_f:
                raw_cache = cache_f.read()
            bool_masks = pickle.loads(raw_cache)

            with open(f"{self.CACHE_PATH}/{cache_key}_target_masks_{purpose}_{gt_batch_idx}.bin", "rb") as cache_f:
                raw_cache = cache_f.read()
            target_masks = pickle.loads(raw_cache)

            for local_slice_idx in range(DATA_BATCH_PER_GT_BATCH):
                slice_idx = local_slice_idx * DATA_LOAD_BATCH_SIZE

                yield bool_masks[0][slice_idx: slice_idx + DATA_LOAD_BATCH_SIZE], \
                        target_masks[0][slice_idx: slice_idx + DATA_LOAD_BATCH_SIZE], \
                        bool_masks[1][slice_idx: slice_idx + DATA_LOAD_BATCH_SIZE], \
                        target_masks[1][slice_idx: slice_idx + DATA_LOAD_BATCH_SIZE], \
                        bool_masks[2][slice_idx: slice_idx + DATA_LOAD_BATCH_SIZE], \
                        target_masks[2][slice_idx: slice_idx + DATA_LOAD_BATCH_SIZE]

        if incomplete is True:

            gt_batch_idx += 1

            print(f"(rem) gt b idx {gt_batch_idx}")

            with open(f"{self.CACHE_PATH}/{cache_key}_bool_masks_{purpose}_{gt_batch_idx}.bin", "rb") as cache_f:
                raw_cache = cache_f.read()
            bool_masks = pickle.loads(raw_cache)

            with open(f"{self.CACHE_PATH}/{cache_key}_target_masks_{purpose}_{gt_batch_idx}.bin", "rb") as cache_f:
                raw_cache = cache_f.read()
            target_masks = pickle.loads(raw_cache)

            rem = IMG_CNT % GT_LOAD_BATCH_SIZE

            slice_idx = 0
            while True:

                if slice_idx + DATA_LOAD_BATCH_SIZE >= rem:
                    yield bool_masks[0][slice_idx:], \
                            target_masks[0][slice_idx:], \
                            bool_masks[1][slice_idx:], \
                            target_masks[1][slice_idx:], \
                            bool_masks[2][slice_idx:], \
                            target_masks[2][slice_idx:]
                    break

                else:
                    yield bool_masks[0][slice_idx: slice_idx + DATA_LOAD_BATCH_SIZE], \
                            target_masks[0][slice_idx: slice_idx + DATA_LOAD_BATCH_SIZE], \
                            bool_masks[1][slice_idx: slice_idx + DATA_LOAD_BATCH_SIZE], \
                            target_masks[1][slice_idx: slice_idx + DATA_LOAD_BATCH_SIZE], \
                            bool_masks[2][slice_idx: slice_idx + DATA_LOAD_BATCH_SIZE], \
                            target_masks[2][slice_idx: slice_idx + DATA_LOAD_BATCH_SIZE]

                    slice_idx += DATA_LOAD_BATCH_SIZE

    def load_data(self, batch_size, purpose):

        # TODO remove in the future for better generalization
        if DATA_LOAD_BATCH_SIZE != batch_size:
            print("Data load batch size must (currently) be equal with the train batch size")
            quit()

        gt_generator = self.load_gt(purpose)
        for imgs in self.load_images(purpose):

            bool_mask_size1, target_mask_size1, bool_mask_size2, target_mask_size2, bool_mask_size3, target_mask_size3 = next(gt_generator)
            
            yield tf.cast(imgs, tf.float32) / 255.0, bool_mask_size1, target_mask_size1, bool_mask_size2, target_mask_size2, bool_mask_size3, target_mask_size3

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

        if self.cache_key is not None:

            try:
                
                with open(f"{self.CACHE_PATH}/{self.cache_key}_anchors.bin", "rb") as cache_f:
                    raw_cache = cache_f.read()

                self.anchors = pickle.loads(raw_cache)

                print("Cache found. Anchors loaded")
                return

            except FileNotFoundError:
                print("Cache not found. Operations will be fully executed and a new cache will be created")

        anchor_finder = AnchorFinder(self.imgs)
        self.anchors = tf.cast(tf.convert_to_tensor(anchor_finder.get_anchors()), tf.int32)

        if self.cache_key is not None:

            new_cache = pickle.dumps(self.anchors)

            with open(f"{self.CACHE_PATH}/{self.cache_key}_anchors.bin", "wb+") as cache_f:
                cache_f.write(new_cache)

    def assign_anchors_to_objects(self):

        if self.used_categories == {}:
            print("info not yet loaded")
            quit()

        if self.anchors == []:
            print("anchors not yet determined")
            quit()

        if self.cache_key is not None:

            try:

                for purpose in ["train", "validation"]:
                    
                    IMG_CNT = len(self.imgs[purpose].keys())
                    GT_BATCH_CNT = IMG_CNT // GT_LOAD_BATCH_SIZE
                    if IMG_CNT % GT_LOAD_BATCH_SIZE > 0:
                        GT_BATCH_CNT += 1

                    for gt_batch_idx in range(GT_BATCH_CNT):

                        with open(f"{self.CACHE_PATH}/{self.cache_key}_bool_masks_{purpose}_{gt_batch_idx}.bin", "rb") as cache_f:
                            pass

                        with open(f"{self.CACHE_PATH}/{self.cache_key}_target_masks_{purpose}_{gt_batch_idx}.bin", "rb") as cache_f:
                            pass

                print("Cache found for ground truth masks. It will be loaded when needed")
                return

            except FileNotFoundError:
                print("Cache not found. Operations will be fully executed and a new cache will be created")

        '''
            if the cache key is none, these values will not be stored for another round
            still, they need to (at least temporarily) reside on the disk
            so, we create a fake cache under FAKE_CACHE_KEY
        '''
        if self.cache_key is None:
            cache_key = DataManager.FAKE_CACHE_KEY
        else:
            cache_key = self.cache_key

        def _iou(anchor, w, h):
            
            intersect_w = np.minimum(anchor[0], w)
            intersect_h = np.minimum(anchor[1], h)

            intersection = intersect_w * intersect_h
            union = w * h + anchor[0] * anchor[1] - intersection

            return intersection / union

        def _store_gt(bool_anchor_masks, target_anchor_masks, purpose, gt_batch_idx):

            for d in range(SCALE_CNT):

                bool_anchor_masks[d] = tf.convert_to_tensor(bool_anchor_masks[d], dtype=tf.float32)
                target_anchor_masks[d] = tf.convert_to_tensor(target_anchor_masks[d], dtype=tf.float32)

            new_cache = pickle.dumps(bool_anchor_masks)

            with open(f"{self.CACHE_PATH}/{cache_key}_bool_masks_{purpose}_{gt_batch_idx}.bin", "wb+") as cache_f:
                cache_f.write(new_cache)

            new_cache = pickle.dumps(target_anchor_masks)

            with open(f"{self.CACHE_PATH}/{cache_key}_target_masks_{purpose}_{gt_batch_idx}.bin", "wb+") as cache_f:
                cache_f.write(new_cache)

        for purpose in ["train", "validation"]:
            
            bool_anchor_masks = [[] for _ in range(SCALE_CNT)]     
            '''
                    for each scale,
                        B x S[scale] x S[scale] x A x 1     - whether that anchor is responsible for an object or not
            '''
            target_anchor_masks = [[] for _ in range(SCALE_CNT)]
            '''
                    for each scale,
                        B x S[scale] x S[scale] x A x 5     - regression targets and the class given by its one_hot index (but NOT one_hot encoded)            
            '''

            gt_batch_idx = 0
            for img_id in self.imgs[purpose].keys():

                bool_mask = []
                target_mask = []

                for d in range(SCALE_CNT):

                    bool_mask.append(np.array([[[[0] for _ in range(ANCHOR_PERSCALE_CNT)] for _ in range(GRID_CELL_CNT[d])] for _ in range(GRID_CELL_CNT[d])], dtype=np.float64))
                    target_mask.append(np.array([[[[0 for _ in range(4 + len(self.category_onehot_to_id))] for _ in range(ANCHOR_PERSCALE_CNT)] for _ in range(GRID_CELL_CNT[d])] for _ in range(GRID_CELL_CNT[d])], dtype=np.float64))

                for bbox_d in self.imgs[purpose][img_id]["objs"]:

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

                    if x == np.floor(x):
                        x -= 0.05

                    if y == np.floor(y):
                        y -= 0.05

                    cx, cy = np.int32(np.floor(x)), np.int32(np.floor(y))
                    x, y = x - cx, y - cy   # x - cx, y - cy == sigmoid(tx) - cx, sigmoid(ty) - cy <=> x = sigmoid(tx), y = sigmoid(ty)

                    # get anchor w and h relative to the grid cell count
                    anchor_w = GRID_CELL_CNT[max_iou_scale] * (self.anchors[max_iou_scale][max_iou_idx][0] / IMG_SIZE[0])
                    anchor_h = GRID_CELL_CNT[max_iou_scale] * (self.anchors[max_iou_scale][max_iou_idx][1] / IMG_SIZE[0])

                    bool_mask[max_iou_scale][cx][cy][max_iou_idx] = np.array([1.0], dtype=np.float64)
                    target_mask[max_iou_scale][cx][cy][max_iou_idx] = np.array(tf.concat([tf.convert_to_tensor([tf.math.log(x / (1 - x))]), 
                                                                                            tf.convert_to_tensor([tf.math.log(y / (1 - y))]),
                                                                                            tf.convert_to_tensor([tf.math.log(w / anchor_w)]), 
                                                                                            tf.convert_to_tensor([tf.math.log(h / anchor_h)]),
                                                                                            tf.cast(tf.one_hot(categ, len(self.category_onehot_to_id)), dtype=tf.double)],
                                                                                            axis=0), dtype=np.float64)

                for d in range(SCALE_CNT):

                    bool_anchor_masks[d].append(tf.convert_to_tensor(bool_mask[d], dtype=tf.float32))
                    target_anchor_masks[d].append(tf.convert_to_tensor(target_mask[d], dtype=tf.float32))

                if len(bool_anchor_masks[0]) == GT_LOAD_BATCH_SIZE:

                    _store_gt(bool_anchor_masks, target_anchor_masks, purpose, gt_batch_idx)

                    bool_anchor_masks = [[] for _ in range(SCALE_CNT)]
                    target_anchor_masks = [[] for _ in range(SCALE_CNT)]
                    gt_batch_idx += 1

            if len(bool_anchor_masks[0]) > 0:
                _store_gt(bool_anchor_masks, target_anchor_masks, purpose, gt_batch_idx)
