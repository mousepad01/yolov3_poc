import json
import zlib
import pickle
import os
import random
import time

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
                                        "objs": [(category one hot index, abs x, abs y, abs w, abs h), ...],
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

        '''
            auxiliary variables
        '''

        self._ans = None
        self._an_area = None
        self._ans_rel = None

    def prepare(self):
        '''
            Prepare everything for detection training/validation/testing, 
            and also for encoder pretraining
        '''

        self.load_info()
        self.determine_anchors()

    def get_img_cnt(self, purpose):

        if purpose == "train":
            return len(self.imgs["train"])

        elif purpose == "validation":
            return len(self.imgs["validation"])

    def get_box_cnt(self, purpose):
        return self._box_cnt[purpose]

    def get_class_cnt(self):
        return len(self.category_onehot_to_id)

    def augment_data(self, image, ground_truth, purpose, img_id):

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

        def _flip_leftright(img, gt):

            '''
                gt: (3x obj mask, 3x ig mask, 3x target mask, gt masks)
            '''

            img = cv.flip(img, 1)

            obj_m = gt[0]
            ign_m = gt[1]
            tar_m = gt[2]
            for d in range(SCALE_CNT):

                obj_m[d] = tf.reverse(obj_m[d], axis=[1])
                ign_m[d] = tf.reverse(ign_m[d], axis=[1])
                tar_m[d] = tf.reverse(tar_m[d], axis=[1])
            
            for d in range(SCALE_CNT):

                reversed_cell_y_offset = 1 - tf.sigmoid(tar_m[d][..., 1:2])
                reversed_ty = tf.math.log(reversed_cell_y_offset / (1 - reversed_cell_y_offset))

                tar_m[d] = tf.concat([tar_m[d][..., 0:1], reversed_ty, tar_m[d][..., 2:]], axis=-1)

            gt_boxes = gt[3]

            reversed_ymin = 1 - gt_boxes[..., 1:2]
            reversed_ymax = 1 - gt_boxes[..., 3:4]

            gt_boxes = tf.concat([gt_boxes[..., 0:1], reversed_ymax, gt_boxes[..., 2:3], reversed_ymin], axis=-1)

            gt_boxes_usedslots = tf.expand_dims(tf.cast(tf.reduce_sum(gt[3], axis=-1) > 0, tf.float32), axis=1)
            gt_boxes *= gt_boxes_usedslots

            return img, (obj_m, ign_m, tar_m, gt_boxes)

        def _shearing(img, gt, purpose, img_id):
            
            sh_x = random.random() / 5
            sh_y = random.random() / 5

            shearing_mat = np.float32( [[1,    sh_y, 0],
                                        [sh_x, 1,    0],
                                        [0,    0,    1]])

            img = cv.warpPerspective(img, shearing_mat, IMG_SIZE)
            
            objs = self.imgs[purpose][img_id]["objs"]
            sheared_objs = []

            for bbox_d in objs:
                
                cat, x, y, w, h = bbox_d
                x, y, w, h = x + sh_x * y, y + sh_y * x, w + sh_x * h, h + sh_y * w
                w, h = min(w, IMG_SIZE[0] - x), min(h, IMG_SIZE[1] - y)

                if x < IMG_SIZE[0] and y < IMG_SIZE[1] and w >= MIN_BBOX_DIM and h >= MIN_BBOX_DIM:
                    sheared_objs.append((cat, x, y, w, h))
                
            return img, self.create_gt(sheared_objs)

        def _rotation(img, gt, purpose, img_id):
            
            alpha = (random.random() * 2 / 3) - 0.33
            cos = np.cos(alpha)
            sin = np.sin(alpha)

            off = IMG_SIZE[0]

            fx = -off * cos + off * sin + off
            fy = -off * sin - off * cos + off

            rotation_mat = np.float32( [[cos, -sin, fx],
                                        [sin,  cos, fy],
                                        [0,    0,   1]])

            img = cv.warpPerspective(img, rotation_mat, IMG_SIZE)

            objs = self.imgs[purpose][img_id]["objs"]
            rotated_objs = []

            for bbox_d in objs:
                
                cat, x, y, w, h = bbox_d

                x0, y0 = x, y
                x3, y3 = x + w, y + h
                x1, y1 = x, y + h
                x2, y2 = x + w, y

                x0_ = y0 * sin + x0 * cos + fy
                y0_ = y0 * cos - x0 * sin + fx

                x1_ = y1 * sin + x1 * cos + fy
                y1_ = y1 * cos - x1 * sin + fx

                x2_ = y2 * sin + x2 * cos + fy
                y2_ = y2 * cos - x2 * sin + fx

                x3_ = y3 * sin + x3 * cos + fy
                y3_ = y3 * cos - x3 * sin + fx

                xmin_ = max(min(x0_, x1_, x2_, x3_), 0)
                ymin_ = max(min(y0_, y1_, y2_, y3_), 0)

                xmax_ = min(max(x0_, x1_, x2_, x3_), IMG_SIZE[0])
                ymax_ = min(max(y0_, y1_, y2_, y3_), IMG_SIZE[0])

                x, y, w, h = xmin_, ymin_, xmax_ - xmin_, ymax_ - ymin_

                # other checks are implicit
                if w >= MIN_BBOX_DIM and h >= MIN_BBOX_DIM:
                    rotated_objs.append((cat, x, y, w, h))

            return img, self.create_gt(rotated_objs)

        # unused
        def _flip_channels(img):
            
            order = [0, 1, 2]
            random.shuffle(order)

            return np.stack([img[..., order[0]], img[..., order[1]], img[..., order[2]]], axis=2)

        image = np.array(image)

        a = random.random()
        if a < 0:#0.33:
            image, ground_truth = _shearing(image, ground_truth, purpose, img_id)
        elif a < 1:#0.66:
            image, ground_truth = _rotation(image, ground_truth, purpose, img_id)

        a = random.random()
        if a < 0:#0.5:
            image, ground_truth = _flip_leftright(image, ground_truth)

        a = random.random()
        if a < 0.5:
            return tf.convert_to_tensor(_blur_contrast(image)), ground_truth
        else:
            return tf.convert_to_tensor(_noise_contrast(image)), ground_truth

    def load_data(self, batch_size, purpose, shuffle=True, augment_probability=0.7):
        '''
            * images get loaded on GPU at the cast when yielding
            * gt gets loaded on GPU at also when loading ???
        '''

        keys = list(self.imgs[purpose].keys())

        if shuffle:
            random.shuffle(keys)

        # FIXME
        yield keys

        current_img_loaded = []
        current_om_loaded = [[] for _ in range(SCALE_CNT)]
        current_im_loaded = [[] for _ in range(SCALE_CNT)]
        current_tm_loaded = [[] for _ in range(SCALE_CNT)]
        current_gtboxes_loaded = []

        for k in keys:

            loaded_img = self.cache_manager.get_img(k, purpose)
            gt = self.cache_manager.get_gt(k, purpose)

            if random.random() < augment_probability:
                loaded_img, gt = self.augment_data(loaded_img, gt, purpose, k)

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

    def augment_pretrain_data(self, image):

        def _blur_contrast(img):

            contrast = 1 + ((random.random() * 2 - 1) / 5)

            img = cv.blur(img, (2, 2))
            img = np.ndarray.astype(np.clip(img * contrast, 0, 255), np.uint8)

            return img

        def _noise_contrast(img):

            noise_mask = NOISE_MASK_POOL[np.random.randint(0, NOISE_MASK_POOL_LEN - 1)][:PRETRAIN_BOX_SIZE[0], :PRETRAIN_BOX_SIZE[1], :]
            contrast = 1 + ((random.random() * 2 - 1) / 5)

            img = np.ndarray.astype(np.clip(img * contrast + noise_mask, 0, 255), np.uint8)

            return img

        def _flip_leftright(img):
            return cv.flip(img, 1)

        # unused
        def _flip_channels(img):
            
            order = [0, 1, 2]
            random.shuffle(order)

            return np.stack([img[..., order[0]], img[..., order[1]], img[..., order[2]]], axis=2)

        image = np.array(image)

        a = random.random()

        if a < 0.5:
            image = _flip_leftright(image)

        a = random.random()

        if a < 0.5:
            return tf.convert_to_tensor(_blur_contrast(image))
        else:
            return tf.convert_to_tensor(_noise_contrast(image))

    def load_pretrain_data(self, batch_size, purpose, shuffle=True, augment_probability=0.7):

        CLASS_CNT = self.get_class_cnt()

        keys = list(self.imgs[purpose].keys())

        if shuffle:
            random.shuffle(keys)
        
        current_loaded = []
        current_gt_loaded = []

        for img_id in keys:
            img_d = self.imgs[purpose][img_id]

            for bbox_d in img_d["objs"]:

                loaded_box = self.cache_manager.get_box(img_id, bbox_d, purpose)

                if random.random() < augment_probability:
                    loaded_box = self.augment_pretrain_data(loaded_box)

                current_loaded.append(loaded_box)
                current_gt_loaded.append(tf.one_hot(bbox_d[0], CLASS_CNT))

                if len(current_loaded) == batch_size:

                    yield (tf.convert_to_tensor(current_loaded, tf.float32) / 255.0) * 2.0 - 1.0, \
                            tf.convert_to_tensor(current_gt_loaded)

                    current_loaded = []
                    current_gt_loaded = []

        if len(current_loaded) > 0:

            yield (tf.convert_to_tensor(current_loaded, tf.float32) / 255.0) * 2.0 - 1.0, \
                    tf.convert_to_tensor(current_gt_loaded)

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
                self.imgs[purpose][anno["image_id"]]["objs"].append((self.used_categories[anno["category_id"]]["onehot"], anno["bbox"][1], 
                                                                        anno["bbox"][0], anno["bbox"][3], anno["bbox"][2]))

            for img_info in info["images"]:

                if img_info["id"] not in self.imgs[purpose]:
                    continue
                
                self.imgs[purpose][img_info["id"]]["filename"] = self.data_path[purpose] + img_info["file_name"]

                objs = self.imgs[purpose][img_info["id"]]["objs"]

                # h, w intentionally inverted
                h = img_info["width"]
                w = img_info["height"]

                off, ratio = _find_resize_factors(w, h)

                if off[0] > 0:

                    for bb_idx in range(len(objs)):

                        cat, x, y, w, h = objs[bb_idx]
                        objs[bb_idx] = (cat, off[0] + x, y, w, h)

                elif off[1] > 0:

                    for bb_idx in range(len(objs)):

                        cat, x, y, w, h = objs[bb_idx]
                        objs[bb_idx] = (cat, x, off[1] + y, w, h)
   
                for bb_idx in range(len(objs)):

                    cat, x, y, w, h = objs[bb_idx]
                    objs[bb_idx] = (cat, np.int32(np.round(ratio * x)), np.int32(np.round(ratio * y)), 
                                            np.int32(np.round(ratio * w)), np.int32(np.round(ratio * h)))

                objs_ok = []
                for bbox_d in objs:

                    if bbox_d[3] < MIN_BBOX_DIM or bbox_d[4] < MIN_BBOX_DIM:
                        continue

                    objs_ok.append(bbox_d)

                self.imgs[purpose][img_info["id"]]["objs"] = objs_ok

                self._box_cnt[purpose] += len(objs_ok)

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

        for purpose in ["train", "validation"]:
            for img_id in self.imgs[purpose].keys():

                self.max_true_boxes = max(len(self.imgs[purpose][img_id]["objs"]), self.max_true_boxes)

        tf.print(f"Loaded {self.get_img_cnt('train')} train images ({self._box_cnt['train']} train boxes) and {self.get_img_cnt('validation')} validation images ({self._box_cnt['validation']} validation boxes).")

    def determine_anchors(self):

        if self.used_categories == {}:
            tf.print("info not yet loaded")
            quit()

        self.cache_manager.get_anchors()
        if self.anchors != []:
            
            self._ans = np.array(self.anchors, dtype=np.float32)
            self._an_area = self._ans[..., 0] * self._ans[..., 1]
            self._ans_rel = self._ans / IMG_SIZE[0]
            self._ans_rel = np.stack([self._ans_rel[0] * GRID_CELL_CNT[0], self._ans_rel[1] * GRID_CELL_CNT[1], self._ans_rel[2] * GRID_CELL_CNT[2]])

            return

        anchor_finder = AnchorFinder(self.imgs)
        self.anchors = tf.cast(tf.convert_to_tensor(anchor_finder.get_anchors()), tf.int32)

        self.cache_manager.store_anchors()

        self._ans = np.array(self.anchors, dtype=np.float32)
        self._an_area = self._ans[..., 0] * self._ans[..., 1]
        self._ans_rel = self._ans / IMG_SIZE[0]
        self._ans_rel = np.stack([self._ans_rel[0] * GRID_CELL_CNT[0], self._ans_rel[1] * GRID_CELL_CNT[1], self._ans_rel[2] * GRID_CELL_CNT[2]])

    def create_gt(self, objs):

        ans = self._ans
        an_area = self._an_area
        ans_rel = self._ans_rel

        obj_mask = []
        ignored_mask = []
        target_mask = []

        for d in range(SCALE_CNT):

            obj_mask.append(np.zeros((GRID_CELL_CNT[d], GRID_CELL_CNT[d], ANCHOR_PERSCALE_CNT, 1), dtype=np.float32))
            ignored_mask.append(np.zeros((GRID_CELL_CNT[d], GRID_CELL_CNT[d], ANCHOR_PERSCALE_CNT, 1), dtype=np.float32))
            target_mask.append(np.zeros((GRID_CELL_CNT[d], GRID_CELL_CNT[d], ANCHOR_PERSCALE_CNT, 5), dtype=np.float32))
    
        for bbox_d in objs:

            categ, x, y, w, h = bbox_d
            categ = np.int32(categ)

            wh = w * h

            max_iou_idx = None
            max_iou_scale = None

            an_int = np.minimum(ans[..., 0], w) * np.minimum(ans[..., 1], h)
            an_union = wh + an_area - an_int

            an_iou = an_int / an_union
            max_an_iou = np.argmax(an_iou)
            max_iou_scale, max_iou_idx = max_an_iou // SCALE_CNT, max_an_iou % ANCHOR_PERSCALE_CNT

            an_iou_ign = an_iou > IGNORED_IOU_THRESHOLD
            ignore_anchors_x, ignore_anchors_y = np.nonzero(an_iou_ign)

            x_, y_, w_, h_ = x + w // 2, y + h // 2, w, h
            x_, y_, w_, h_ = x_ / IMG_SIZE[0], y_ / IMG_SIZE[1], w_ / IMG_SIZE[0], h_ / IMG_SIZE[1]
            x_, y_, w_, h_ = x_ * GRID_CELL_CNT[max_iou_scale], y_ * GRID_CELL_CNT[max_iou_scale], w_ * GRID_CELL_CNT[max_iou_scale], h_ * GRID_CELL_CNT[max_iou_scale]

            if x_ == np.floor(x_):
                x_ -= 0.05

            if y_ == np.floor(y_):
                y_ -= 0.05

            cx, cy = np.int32(np.floor(x_)), np.int32(np.floor(y_))
            x_, y_ = x_ - cx, y_ - cy 

            anchor_w = ans_rel[max_iou_scale, max_iou_idx, 0]
            anchor_h = ans_rel[max_iou_scale, max_iou_idx, 1]

            obj_mask[max_iou_scale][cx][cy][max_iou_idx][0] = 1.0
            target_mask[max_iou_scale][cx][cy][max_iou_idx][0] = np.log(x_ / (1 - x_))
            target_mask[max_iou_scale][cx][cy][max_iou_idx][1] = np.log(y_ / (1 - y_))
            target_mask[max_iou_scale][cx][cy][max_iou_idx][2] = np.log(w_ / anchor_w)
            target_mask[max_iou_scale][cx][cy][max_iou_idx][3] = np.log(h_ / anchor_h)
            target_mask[max_iou_scale][cx][cy][max_iou_idx][4] = categ

            x, y = x + w // 2, y + h // 2
            x, y = x / IMG_SIZE[0], y / IMG_SIZE[1]

            for ign_idx in range(ignore_anchors_x.shape[0]):

                a_scale, a_idx = ignore_anchors_x[ign_idx], ignore_anchors_y[ign_idx]

                x_, y_ = x * GRID_CELL_CNT[a_scale], y * GRID_CELL_CNT[a_scale]
                if x_ == np.floor(x_):
                    x_ -= 0.05
                if y_ == np.floor(y_):
                    y_ -= 0.05

                cx, cy = np.int32(np.floor(x_)), np.int32(np.floor(y_))
                
                ignored_mask[a_scale][cx][cy][a_idx][0] = 1

        for d in range(SCALE_CNT):
            ignored_mask[d] = np.logical_and(obj_mask[d] == 0, ignored_mask[d] == 1)

        for d in range(SCALE_CNT):

            obj_mask[d] = tf.convert_to_tensor(obj_mask[d], dtype=tf.float32)
            ignored_mask[d] = tf.convert_to_tensor(ignored_mask[d], dtype=tf.float32)
            target_mask[d] = tf.convert_to_tensor(target_mask[d], dtype=tf.float32)

        gt_boxes = []
        for bbox_d in objs:

            _, x, y, w, h = bbox_d
            xmin, ymin, xmax, ymax = x, y, x + w, y + h
            xmin, ymin, xmax, ymax = xmin / IMG_SIZE[0], ymin / IMG_SIZE[1], xmax / IMG_SIZE[0], ymax / IMG_SIZE[1]

            gt_boxes.append([xmin, ymin, xmax, ymax])

        for _ in range(len(gt_boxes), self.max_true_boxes, 1):
            gt_boxes.append([0, 0, 0, 0])

        gt_boxes = tf.convert_to_tensor(gt_boxes, dtype=tf.float32)

        return (obj_mask, ignored_mask, target_mask, gt_boxes)

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

            box = img[coords[1]: coords[1] + coords[3], coords[2]: coords[2] + coords[4], :]
            box = self.resize_with_pad(box, PRETRAIN_BOX_SIZE)

            if len(self._permanent_pretrain_data["train"]) + len(self._permanent_pretrain_data["validation"]) < PERMANENT_PRETRAIN_DATA_ENTRIES:
                
                if COMPRESS_DATA_CACHE:
                    _, encoded_box = cv.imencode(".jpg", box)
                    self._permanent_pretrain_data[purpose][entry_key] = encoded_box
                else:
                    self._permanent_pretrain_data[purpose][entry_key] = tf.convert_to_tensor(box)

            return tf.convert_to_tensor(box)

    def get_gt(self, img_id, purpose):

        if img_id in self._permanent_gt.keys():

            if COMPRESS_GT_CACHE_LEVEL != 0:
                return (pickle.loads(zlib.decompress(self._permanent_gt[img_id][0])), \
                        pickle.loads(zlib.decompress(self._permanent_gt[img_id][1])), \
                        pickle.loads(zlib.decompress(self._permanent_gt[img_id][2])),
                        self._permanent_gt[img_id][3])

            else:
                return self._permanent_gt[img_id]

        else:

            obj_masks, ignored_masks, target_masks, gt_boxes = self.loader.create_gt(self.loader.imgs[purpose][img_id]["objs"])

            if len(self._permanent_gt) < PERMANENT_DATA_ENTRIES:

                if COMPRESS_GT_CACHE_LEVEL != 0:
                    self._permanent_gt[img_id] = (zlib.compress(pickle.dumps(obj_masks), COMPRESS_GT_CACHE_LEVEL), \
                                                    zlib.compress(pickle.dumps(ignored_masks), COMPRESS_GT_CACHE_LEVEL), \
                                                    zlib.compress(pickle.dumps(target_masks), COMPRESS_GT_CACHE_LEVEL),
                                                    gt_boxes)
                else:
                    self._permanent_gt[img_id] = (obj_masks, ignored_masks, target_masks, gt_boxes)

            return (obj_masks, ignored_masks, target_masks, gt_boxes)
