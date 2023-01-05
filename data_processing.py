import json
import zlib
import pickle
import os
import random

import cv2 as cv
import numpy as np
import tensorflow as tf

from anchor_kmeans import *
from constants import *

class DataLoader:

    def __init__(self, train_data_path=TRAIN_DATA_PATH,
                        train_info_path=TRAIN_INFO_PATH,
                        validation_data_path=VALIDATION_DATA_PATH,
                        validation_info_path=VALIDATION_INFO_PATH,

                        cache_key=None,
                        superclasses=["person", "vehicle", "outdoor", "animal", "accessory", \
                                        "sports", "kitchen", "food", "furniture", "electronic", \
                                        "appliance", "indoor"],
                        classes=[],
                        validation_ratio=None,
                    ):

        assert(cache_key != TMP_CACHE_KEY)
        
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

        self.classes = classes
        '''
            to specify a data subset at load_info()
        '''

        self.superclasses = superclasses
        '''
            to specify a data subset at load_info()
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
                                        "filename": complete file path (relative to this project's path)
                                    }
        '''

        self._box_cnt = {"train": 0, "validation": 0}
        '''
            how many boxes for train / validation (used only for pre-training)
        '''

        self.max_true_boxes = 0
        '''
            maximum true boxes count per image
        '''

        self.validation_ratio = validation_ratio
        '''
            (approximate) ratio of |{val imgs}| / |{all imgs}|
            * to keep original distribution, whatever it may be, call with validation_ratio=None
        '''

        self.anchors = []
        '''
            D x A x 2 (D = SCALE_COUNT)
            * cacheable
        '''

        self.cache_manager = DataCacheManager(self, cache_key)
        '''
            the cache manager
        '''

    def prepare(self):
        '''
            Prepare everything for detection training/validation/testing, 
            and also for encoder pretraining
        '''

        self.load_info()
        self.determine_anchors()
        self.assign_anchors_to_objects()

    def get_img_cnt(self, purpose):

        if purpose == "train":
            return len(self.imgs["train"])

        elif purpose == "validation":
            return len(self.imgs["validation"])

    def get_box_cnt(self, purpose):
        return self._box_cnt[purpose]

    def get_class_cnt(self):
        return len(self.category_onehot_to_id)

    def augment_data(self, image, ground_truth):

        def _blur_contrast(img):

            contrast = 1 + ((random.random() * 2 - 1) / 5)

            img = cv.blur(img, (2, 2))
            img = np.ndarray.astype(np.clip(img * contrast, 0, 255), np.uint8)

            return img

        def _noise_contrast(img):

            noise_mask = NOISE_MASK_POOL[np.random.randint(0, NOISE_MASK_POOL_LEN - 1)]
            contrast = 1 + ((random.random() * 2 - 1) / 5)

            img = np.ndarray.astype(np.clip(img * contrast + noise_mask, 0, 255), np.uint8)

            return img

        # FIXME
        def _flip_leftright(img, gt):

            img = cv.flip(img, 1)

            #gt = tf.reverse(gt, )

            return img, gt

        '''a = random.random()

        if a < 0.5:
            image, ground_truth = _flip_leftright(image, ground_truth)'''

        a = random.random()

        if a < 0.5:
            return tf.convert_to_tensor(_blur_contrast(np.array(image))), ground_truth
        else:
            return tf.convert_to_tensor(_noise_contrast(np.array(image))), ground_truth

    def load_data(self, batch_size, purpose, shuffle=True):
        '''
            * images get loaded on GPU at the cast when yielding
            * gt gets loaded on GPU at also when loading ???
        '''

        keys = list(self.imgs[purpose].keys())

        if shuffle:
            random.shuffle(keys)

        # FIXME
        #yield keys

        current_img_loaded = []
        current_om_loaded = [[] for _ in range(SCALE_CNT)]
        current_im_loaded = [[] for _ in range(SCALE_CNT)]
        current_tm_loaded = [[] for _ in range(SCALE_CNT)]
        current_gtboxes_loaded = []

        for k in keys:

            loaded_img = self.cache_manager.get_img(k, purpose)
            gt = self.cache_manager.get_gt(k)

            #tf.print(loaded_img.dtype)
            #tf.print(loaded_img.device)

            if random.random() < AUGMENT_DATA_PROBABILITY:
                loaded_img, gt = self.augment_data(loaded_img, gt)

            #tf.print(loaded_img.dtype)
            #tf.print(loaded_img.device)

            #quit()

            current_img_loaded.append(loaded_img)

            for d in range(SCALE_CNT):

                current_om_loaded[d].append(gt[0][d])
                current_im_loaded[d].append(gt[1][d])
                current_tm_loaded[d].append(gt[2][d])

            current_gtboxes_loaded.append(gt[3])

            if len(current_img_loaded) == batch_size:

                yield (tf.convert_to_tensor(current_img_loaded, dtype=tf.float32) / 255.0) * 2.0 - 1.0, \
                        tf.convert_to_tensor(current_om_loaded[0]), \
                        tf.convert_to_tensor(current_im_loaded[0]), \
                        tf.convert_to_tensor(current_tm_loaded[0]), \
                        tf.convert_to_tensor(current_om_loaded[1]), \
                        tf.convert_to_tensor(current_im_loaded[1]), \
                        tf.convert_to_tensor(current_tm_loaded[1]), \
                        tf.convert_to_tensor(current_om_loaded[2]), \
                        tf.convert_to_tensor(current_im_loaded[2]), \
                        tf.convert_to_tensor(current_tm_loaded[2]), \
                        tf.reshape(tf.convert_to_tensor(current_gtboxes_loaded), (batch_size, 1, 1, 1, -1, 4))

                current_img_loaded = []
                current_om_loaded = [[] for _ in range(SCALE_CNT)]
                current_im_loaded = [[] for _ in range(SCALE_CNT)]
                current_tm_loaded = [[] for _ in range(SCALE_CNT)]
                current_gtboxes_loaded = []

        if len(current_img_loaded) > 0:

            yield (tf.convert_to_tensor(current_img_loaded, dtype=tf.float32) / 255.0) * 2.0 - 1.0, \
                    tf.convert_to_tensor(current_om_loaded[0]), \
                    tf.convert_to_tensor(current_im_loaded[0]), \
                    tf.convert_to_tensor(current_tm_loaded[0]), \
                    tf.convert_to_tensor(current_om_loaded[1]), \
                    tf.convert_to_tensor(current_im_loaded[1]), \
                    tf.convert_to_tensor(current_tm_loaded[1]), \
                    tf.convert_to_tensor(current_om_loaded[2]), \
                    tf.convert_to_tensor(current_im_loaded[2]), \
                    tf.convert_to_tensor(current_tm_loaded[2]), \
                    tf.reshape(tf.convert_to_tensor(current_gtboxes_loaded), (len(current_img_loaded), 1, 1, 1, -1, 4))

    def load_boxes(self, purpose):
        
        current_loaded = []
        for img_id, img_d in self.imgs[purpose].items():
            for bbox_d in img_d["objs"]:

                current_loaded.append(self.cache_manager.get_box(img_id, bbox_d["bbox"], purpose))

                if len(current_loaded) == PRETRAIN_DATA_LOAD_BATCH_SIZE:

                    current_loaded = tf.convert_to_tensor(current_loaded)
                    yield current_loaded

                    current_loaded = []

        if len(current_loaded) > 0:

            current_loaded = tf.convert_to_tensor(current_loaded)
            yield current_loaded

    def load_box_gt(self, purpose):

        CLASS_CNT = self.get_class_cnt()
        
        current_loaded = []
        for _, img_d in self.imgs[purpose].items():
            for bbox_d in img_d["objs"]:

                current_loaded.append(tf.one_hot(bbox_d["category"], CLASS_CNT))

                if len(current_loaded) == PRETRAIN_GT_LOAD_BATCH_SIZE:

                    current_loaded = tf.convert_to_tensor(current_loaded)
                    yield current_loaded

                    current_loaded = []

        if len(current_loaded) > 0:

            current_loaded = tf.convert_to_tensor(current_loaded)
            yield current_loaded

    def load_pretrain_data(self, batch_size, purpose):
        
        if PRETRAIN_DATA_LOAD_BATCH_SIZE < batch_size:
            tf.print("(Pretrain) Data load batch size must be >= with the train batch size")
            quit()

        if PRETRAIN_DATA_LOAD_BATCH_SIZE % batch_size != 0:
            tf.print("(Pretrain) Data load batch size must be divisible with the train batch size")
            quit()

        load_2_b = PRETRAIN_DATA_LOAD_BATCH_SIZE // batch_size

        gt_generator = self.load_box_gt(purpose)
        for boxes in self.load_boxes(purpose):

            classif_gt = next(gt_generator)

            if boxes.shape[0] == PRETRAIN_DATA_LOAD_BATCH_SIZE:

                idx = 0
                for idx in range(load_2_b):
                    lo = idx * batch_size
                    hi = (idx + 1) * batch_size

                    yield (tf.cast(boxes[lo: hi], tf.float32) / 255.0) * 2.0 - 1.0, classif_gt[lo: hi]
            
            else:

                limit = boxes.shape[0] // batch_size

                idx = 0
                for idx in range(limit):
                    lo = idx * batch_size
                    hi = (idx + 1) * batch_size

                    yield (tf.cast(boxes[lo: hi], tf.float32) / 255.0) * 2.0 - 1.0, classif_gt[lo: hi]

                lo = limit * batch_size
                yield (tf.cast(boxes[lo:], tf.float32) / 255.0) * 2.0 - 1.0, classif_gt[lo:]

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
                ratio = IMG_SIZE[0] / h

            elif h < w:

                q1 = (w - h) // 2

                off = (0, q1)
                ratio = IMG_SIZE[0] / w

            else:

                off = (0, 0)
                ratio = IMG_SIZE[0] / w    # or h

            return off, ratio

        for purpose in ["train", "validation"]:

            with open(self.info_path[purpose], "r") as info_f:

                info = info_f.read()
                info = json.loads(info)

            if self.used_categories == {}:
                for categ in info["categories"]:

                    if (categ["name"] not in self.classes) and \
                        (categ["supercategory"] not in self.superclasses):

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
                
                self.imgs[purpose][img_info["id"]]["filename"] = self.data_path[purpose] + img_info["file_name"]

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
                    bbox_d["bbox"] = (np.int32(np.round(ratio * bbox_d["bbox"][0])), np.int32(np.round(ratio * bbox_d["bbox"][1])), 
                                        np.int32(np.round(ratio * bbox_d["bbox"][2])), np.int32(np.round(ratio * bbox_d["bbox"][3])))

                bbox_d_ok = []
                for bbox_d in self.imgs[purpose][img_info["id"]]["objs"]:

                    if bbox_d["bbox"][2] < MIN_BBOX_DIM or bbox_d["bbox"][3] < MIN_BBOX_DIM:
                        continue

                    bbox_d_ok.append(bbox_d)

                self.imgs[purpose][img_info["id"]]["objs"] = bbox_d_ok

                self._box_cnt[purpose] += len(bbox_d_ok)

        if self.validation_ratio:

            t_img_cnt = self.get_img_cnt('train')
            v_img_cnt = self.get_img_cnt('validation')

            v_raw_ratio = v_img_cnt / (v_img_cnt + t_img_cnt)

            relocate_img_cnt = (v_img_cnt * self.validation_ratio) // v_raw_ratio
            relocate_img_cnt -= v_img_cnt

            if relocate_img_cnt < 0:

                p_from = "validation"
                p_to = "train"
                relocate_img_cnt *= -1

            elif relocate_img_cnt > 0:
                
                p_from = "train"
                p_to = "validation"

            if relocate_img_cnt > 0:

                reloc_ids = []

                for img_id in self.imgs[p_from].keys():
                    reloc_ids.append(img_id)

                    if len(reloc_ids) == relocate_img_cnt:
                        break

                for img_id in reloc_ids:

                    self.imgs[p_to][img_id] = self.imgs[p_from].pop(img_id)

                    self._box_cnt[p_to] += len(self.imgs[p_to][img_id]["objs"])
                    self._box_cnt[p_from] -= len(self.imgs[p_to][img_id]["objs"])

        tf.print(f"Loaded {self.get_img_cnt('train')} train images ({self._box_cnt['train']} train boxes) and {self.get_img_cnt('validation')} validation images ({self._box_cnt['validation']} validation boxes).")

    def determine_anchors(self):

        if self.used_categories == {}:
            tf.print("info not yet loaded")
            quit()

        self.cache_manager.get_anchors()
        if self.anchors != []:
            return

        anchor_finder = AnchorFinder(self.imgs)
        self.anchors = tf.cast(tf.convert_to_tensor(anchor_finder.get_anchors()), tf.int32)

        self.cache_manager.store_anchors()

    def assign_anchors_to_objects(self):

        if self.used_categories == {}:
            tf.print("info not yet loaded")
            quit()

        if self.anchors == []:
            tf.print("anchors not yet determined")
            quit()

        if self.cache_manager.check_gt():
            return

        def _iou(anchor, w, h):
            
            intersect_w = np.minimum(anchor[0], w)
            intersect_h = np.minimum(anchor[1], h)

            intersection = intersect_w * intersect_h
            union = w * h + anchor[0] * anchor[1] - intersection

            return intersection / union

        for purpose in ["train", "validation"]:
            for img_id in self.imgs[purpose].keys():

                self.max_true_boxes = max(len(self.imgs[purpose][img_id]["objs"]), self.max_true_boxes)

        for purpose in ["train", "validation"]:
            
            for img_id in self.imgs[purpose].keys():

                obj_mask = []
                ignored_mask = []
                target_mask = []

                for d in range(SCALE_CNT):

                    obj_mask.append(np.array([[[[0] for _ in range(ANCHOR_PERSCALE_CNT)] for _ in range(GRID_CELL_CNT[d])] for _ in range(GRID_CELL_CNT[d])], dtype=np.float64))
                    ignored_mask.append(np.array([[[[0] for _ in range(ANCHOR_PERSCALE_CNT)] for _ in range(GRID_CELL_CNT[d])] for _ in range(GRID_CELL_CNT[d])], dtype=np.float64))
                    target_mask.append(np.array([[[[0 for _ in range(5)] for _ in range(ANCHOR_PERSCALE_CNT)] for _ in range(GRID_CELL_CNT[d])] for _ in range(GRID_CELL_CNT[d])], dtype=np.float64))

                for bbox_d in self.imgs[purpose][img_id]["objs"]:

                    categ = np.int32(bbox_d["category"])
                    x, y, w, h = bbox_d["bbox"]

                    max_iou = -1
                    max_iou_idx = None
                    max_iou_scale = None

                    ignore_anchors = []

                    for d in range(SCALE_CNT):
                        for a in range(ANCHOR_PERSCALE_CNT):

                            current_iou = _iou(self.anchors[d][a], w, h)

                            if current_iou > max_iou:
                                
                                max_iou = current_iou
                                max_iou_idx = a
                                max_iou_scale = d

                            if current_iou >= IGNORED_IOU_THRESHOLD:
                                ignore_anchors.append((a, d))

                    x_, y_, w_, h_ = x + w // 2, y + h // 2, w, h
                    x_, y_, w_, h_ = x_ / IMG_SIZE[0], y_ / IMG_SIZE[1], w_ / IMG_SIZE[0], h_ / IMG_SIZE[1]
                    x_, y_, w_, h_ = x_ * GRID_CELL_CNT[max_iou_scale], y_ * GRID_CELL_CNT[max_iou_scale], w_ * GRID_CELL_CNT[max_iou_scale], h_ * GRID_CELL_CNT[max_iou_scale]

                    if x_ == np.floor(x_):
                        x_ -= 0.05

                    if y_ == np.floor(y_):
                        y_ -= 0.05

                    cx, cy = np.int32(np.floor(x_)), np.int32(np.floor(y_))
                    x_, y_ = x_ - cx, y_ - cy 

                    anchor_w = GRID_CELL_CNT[max_iou_scale] * (self.anchors[max_iou_scale][max_iou_idx][0] / IMG_SIZE[0])
                    anchor_h = GRID_CELL_CNT[max_iou_scale] * (self.anchors[max_iou_scale][max_iou_idx][1] / IMG_SIZE[1])

                    obj_mask[max_iou_scale][cx][cy][max_iou_idx] = [1.0]
                    target_mask[max_iou_scale][cx][cy][max_iou_idx] = np.array(tf.concat([tf.convert_to_tensor([tf.math.log(x_ / (1 - x_))]), 
                                                                                            tf.convert_to_tensor([tf.math.log(y_ / (1 - y_))]),
                                                                                            tf.convert_to_tensor([tf.math.log(w_ / anchor_w)]), 
                                                                                            tf.convert_to_tensor([tf.math.log(h_ / anchor_h)]),
                                                                                            tf.convert_to_tensor([categ], dtype=tf.double)],
                                                                                            axis=0), dtype=np.float64)

                    if tf.reduce_sum(tf.cast(tf.math.is_nan(target_mask[max_iou_scale][cx][cy][max_iou_idx]), tf.int32)) > 0:
                        tf.print("Nan found when assigning anchors; possible error?")
                        quit()

                    if tf.reduce_sum(tf.cast(tf.math.is_inf(target_mask[max_iou_scale][cx][cy][max_iou_idx]), tf.int32)) > 0:
                        tf.print("Inf found when assigning anchors; try to make MIN_BBOX_DIM bigger.")
                        quit()

                    for a_idx, a_scale in ignore_anchors:
                        
                        x_, y_ = x + w // 2, y + h // 2
                        x_, y_ = x_ / IMG_SIZE[0], y_ / IMG_SIZE[1]
                        x_, y_ = x_ * GRID_CELL_CNT[a_scale], y_ * GRID_CELL_CNT[a_scale]

                        if x_ == np.floor(x_):
                            x_ -= 0.05

                        if y_ == np.floor(y_):
                            y_ -= 0.05

                        cx, cy = np.int32(np.floor(x_)), np.int32(np.floor(y_))
                        
                        ignored_mask[a_scale][cx][cy][a_idx] = [1]

                for d in range(SCALE_CNT):
                    for cx in range(GRID_CELL_CNT[d]):
                        for cy in range(GRID_CELL_CNT[d]):
                            for a_idx in range(ANCHOR_PERSCALE_CNT):

                                if (obj_mask[d][cx][cy][a_idx] == [0]) and (ignored_mask[d][cx][cy][a_idx] == [1]):
                                    ignored_mask[d][cx][cy][a_idx] = [1.0]
                                else:
                                    ignored_mask[d][cx][cy][a_idx] = [0.0]

                for d in range(SCALE_CNT):

                    obj_mask[d] = tf.convert_to_tensor(obj_mask[d], dtype=tf.float32)
                    ignored_mask[d] = tf.convert_to_tensor(ignored_mask[d], dtype=tf.float32)
                    target_mask[d] = tf.convert_to_tensor(target_mask[d], dtype=tf.float32)

                gt_boxes = []
                for bbox_d in self.imgs[purpose][img_id]["objs"]:

                    x, y, w, h = bbox_d["bbox"]
                    xmin, ymin, xmax, ymax = x, y, x + w, y + h
                    xmin, ymin, xmax, ymax = xmin / IMG_SIZE[0], ymin / IMG_SIZE[1], xmax / IMG_SIZE[0], ymax / IMG_SIZE[1]

                    gt_boxes.append([xmin, ymin, xmax, ymax])

                for _ in range(len(gt_boxes), self.max_true_boxes, 1):
                    gt_boxes.append([0, 0, 0, 0])

                assert(len(gt_boxes) == self.max_true_boxes)
                gt_boxes = tf.convert_to_tensor(gt_boxes, dtype=tf.float32)

                self.cache_manager.store_gt(obj_mask, ignored_mask, target_mask, gt_boxes, img_id)

    def test_for_nan_inf(self):

        for purpose in ["train", "validation"]:
            for (_, obj_mask_size1, ignored_mask_size1, target_mask_size1, \
                    obj_mask_size2, ignored_mask_size2, target_mask_size2, \
                    obj_mask_size3, ignored_mask_size3, target_mask_size3) in self.load_data(128, purpose):  

                assert(tf.reduce_sum(tf.cast(tf.math.is_nan(obj_mask_size1), tf.int32)) == 0)
                assert(tf.reduce_sum(tf.cast(tf.math.is_nan(obj_mask_size2), tf.int32)) == 0)
                assert(tf.reduce_sum(tf.cast(tf.math.is_nan(obj_mask_size3), tf.int32)) == 0)
                assert(tf.reduce_sum(tf.cast(tf.math.is_inf(obj_mask_size1), tf.int32)) == 0)
                assert(tf.reduce_sum(tf.cast(tf.math.is_inf(obj_mask_size2), tf.int32)) == 0)
                assert(tf.reduce_sum(tf.cast(tf.math.is_inf(obj_mask_size3), tf.int32)) == 0)
                assert(tf.reduce_sum(tf.cast(tf.math.is_nan(ignored_mask_size1), tf.int32)) == 0)
                assert(tf.reduce_sum(tf.cast(tf.math.is_nan(ignored_mask_size2), tf.int32)) == 0)
                assert(tf.reduce_sum(tf.cast(tf.math.is_nan(ignored_mask_size3), tf.int32)) == 0)
                assert(tf.reduce_sum(tf.cast(tf.math.is_inf(ignored_mask_size1), tf.int32)) == 0)
                assert(tf.reduce_sum(tf.cast(tf.math.is_inf(ignored_mask_size2), tf.int32)) == 0)
                assert(tf.reduce_sum(tf.cast(tf.math.is_inf(ignored_mask_size3), tf.int32)) == 0)
                assert(tf.reduce_sum(tf.cast(tf.math.is_nan(target_mask_size1), tf.int32)) == 0)
                assert(tf.reduce_sum(tf.cast(tf.math.is_nan(target_mask_size2), tf.int32)) == 0)
                assert(tf.reduce_sum(tf.cast(tf.math.is_nan(target_mask_size3), tf.int32)) == 0)
                assert(tf.reduce_sum(tf.cast(tf.math.is_inf(target_mask_size1), tf.int32)) == 0)
                assert(tf.reduce_sum(tf.cast(tf.math.is_inf(target_mask_size2), tf.int32)) == 0)
                assert(tf.reduce_sum(tf.cast(tf.math.is_inf(target_mask_size3), tf.int32)) == 0)

                assert(tf.reduce_sum(tf.cast(obj_mask_size1 * ignored_mask_size1, tf.int32)) == 0)
                assert(tf.reduce_sum(tf.cast(obj_mask_size2 * ignored_mask_size2, tf.int32)) == 0)
                assert(tf.reduce_sum(tf.cast(obj_mask_size3 * ignored_mask_size3, tf.int32)) == 0)

class DataCacheManager:

    def __init__(self, loader: DataLoader, cache_key):

        self.loader = loader
        '''
            the data_loader object which calls the constructor
        '''

        self.cache_key = cache_key
        '''
            * determining anchors and assigning them to each bounding box can be done only once for a specific dataset
                * if cache_key is given and there is such cache, it upload data cached under this key when specific operations are called; 
                * if cache_key is given but there is no such cache, it will create it;
            * the cache key is also used for model saving/loading, if opted for it when declaring the model
        '''

        self._permanent_data = {}
        '''
            self._permanent_data[img_id] = img entry ready2use w/o loading
        '''

        self._permanent_gt = {}
        '''
            self._permanent_gt[img_id] = gt entry ready2use w/o loading
        '''

        self._permanent_pretrain_data = {"train": {}, "validation": {}}
        '''
            self._permanent_pretrain_data[purpose][(img_id, coords)] = box ready2use w/o loading
        '''
    
    def get_anchors(self):

        if self.cache_key is not None:

            if os.path.exists(f"{DATA_CACHE_PATH}{self.cache_key}/"):

                with open(f"{DATA_CACHE_PATH}{self.cache_key}/anchors.bin", "rb") as cache_f:
                    raw_cache = cache_f.read()

                self.loader.anchors = pickle.loads(raw_cache)
                
                tf.print("Cache found. Anchors loaded.")
                return
            
            os.mkdir(f"{DATA_CACHE_PATH}{self.cache_key}/")

            tf.print("Anchor cache not found. Operations will be fully executed and a new cache will be created")

    def store_anchors(self):

        if self.cache_key is not None:

            new_cache = pickle.dumps(self.loader.anchors)

            with open(f"{DATA_CACHE_PATH}{self.cache_key}/anchors.bin", "wb+") as cache_f:
                cache_f.write(new_cache)

    def resize_with_pad(self, img, final_size):
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

        return cv.resize(img, final_size)

    def get_img(self, img_id, purpose):

        if img_id in self._permanent_data.keys():

            if COMPRESS_DATA_CACHE:
                return tf.convert_to_tensor(cv.imdecode(self._permanent_data[img_id], cv.IMREAD_UNCHANGED))
            else:
                return self._permanent_data[img_id]

        else:

            img = cv.imread(self.loader.imgs[purpose][img_id]["filename"])
            img = self.resize_with_pad(img, IMG_SIZE)

            if len(self._permanent_data) < PERMANENT_DATA_ENTRIES:

                if COMPRESS_DATA_CACHE:
                    _, encoded_img = cv.imencode(".jpg", img)
                    self._permanent_data[img_id] = encoded_img
                else:
                    self._permanent_data[img_id] = tf.convert_to_tensor(img)

            return tf.convert_to_tensor(img)

    def get_box(self, img_id, coords, purpose):
        
        entry_key = (img_id, coords)

        if entry_key in self._permanent_pretrain_data[purpose].keys():

            if COMPRESS_DATA_CACHE:
                return tf.convert_to_tensor(cv.imdecode(self._permanent_pretrain_data[purpose][entry_key], cv.IMREAD_UNCHANGED))
            else:
                return self._permanent_pretrain_data[purpose][entry_key]

        else:

            img = np.array(self.get_img(img_id, purpose))

            box = img[coords[0]: coords[0] + coords[2], coords[1]: coords[1] + coords[3], :]
            box = self.resize_with_pad(box, PRETRAIN_BOX_SIZE)

            if len(self._permanent_pretrain_data["train"]) + len(self._permanent_pretrain_data["validation"]) < PERMANENT_PRETRAIN_DATA_ENTRIES:
                
                if COMPRESS_DATA_CACHE:
                    _, encoded_box = cv.imencode(".jpg", box)
                    self._permanent_pretrain_data[purpose][entry_key] = encoded_box
                else:
                    self._permanent_pretrain_data[purpose][entry_key] = tf.convert_to_tensor(box)

            return tf.convert_to_tensor(box)

    def get_gt(self, img_id):

        if img_id in self._permanent_gt.keys():

            if COMPRESS_GT_CACHE_LEVEL != 0:
                return (pickle.loads(zlib.decompress(self._permanent_gt[img_id][0])), \
                        pickle.loads(zlib.decompress(self._permanent_gt[img_id][1])), \
                        pickle.loads(zlib.decompress(self._permanent_gt[img_id][2])),
                        self._permanent_gt[img_id][3])

            else:
                return self._permanent_gt[img_id]

        else:

            if self.cache_key is None:
                cache_key = TMP_CACHE_KEY
            else:
                cache_key = self.cache_key

            with open(f"{DATA_CACHE_PATH}{cache_key}/gt_{img_id}.bin", "rb") as cache_f:
                raw_cache = cache_f.read()
            gt = pickle.loads(zlib.decompress(raw_cache))
            obj_masks, ignored_masks, target_masks, gt_boxes = gt

            if len(self._permanent_gt) < PERMANENT_DATA_ENTRIES:

                if COMPRESS_GT_CACHE_LEVEL != 0:
                    self._permanent_gt[img_id] = (zlib.compress(pickle.dumps(obj_masks), COMPRESS_GT_CACHE_LEVEL), \
                                                    zlib.compress(pickle.dumps(ignored_masks), COMPRESS_GT_CACHE_LEVEL), \
                                                    zlib.compress(pickle.dumps(target_masks), COMPRESS_GT_CACHE_LEVEL),
                                                    gt_boxes)
                else:
                    self._permanent_gt[img_id] = (obj_masks, ignored_masks, target_masks, gt_boxes)

            return (obj_masks, ignored_masks, target_masks, gt_boxes)

    def check_gt(self):

        if self.cache_key is not None:

            if len(os.listdir(f"{DATA_CACHE_PATH}{self.cache_key}/")) > 1:

                tf.print("Cache found for ground truth masks. It will be loaded when needed")
                return True

            tf.print("Ground truth cache not found. Operations will be fully executed and a new cache will be created")
            return False
        
        return False

    def store_gt(self, obj_gt, ignored_gt, target_gt, gt_boxes, img_id):

        if self.cache_key is None:
            cache_key = TMP_CACHE_KEY
        else:
            cache_key = self.cache_key

        gt = (obj_gt, ignored_gt, target_gt, gt_boxes)

        with open(f"{DATA_CACHE_PATH}{cache_key}/gt_{img_id}.bin", "wb+") as cache_f:
            cache_f.write(zlib.compress(pickle.dumps(gt), 1))
