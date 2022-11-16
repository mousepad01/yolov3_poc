import numpy as np
import tensorflow as tf
import cv2 as cv
import json

from anchor_kmeans import *

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

    DATA_LOAD_BATCH_SIZE = 128
    IMG_SIZE = (416, 416)

    GRID_CELL_COUNT = [13, 26, 52]
    '''
        for each scale, the value of S
    '''

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
        
        self.imgs = {
                        "train": {},
                        "validation": {}
                    }

        self.anchors = []

        self.data_load_batch_size = data_load_batch_size

        self.img_size = img_size

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
            img = cv.resize(img, self.img_size)

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
                                                            "supercategory": categ["supercategory"]
                                                        }

            for anno in info["annotations"]:
                
                if anno["category_id"] not in self.used_categories:
                    continue
                
                if anno["image_id"] not in self.imgs[purpose].keys():
                    self.imgs[purpose][anno["image_id"]] = {
                                                                "objs": [],
                                                                "filename": ""
                                                            }

                self.imgs[purpose][anno["image_id"]]["objs"].append({
                                                                        "category_id": anno["category_id"],
                                                                        "bbox": anno["bbox"]
                                                                    })
            
            for img_info in info["images"]:

                if img_info["id"] not in self.imgs[purpose]:
                    continue
                
                self.imgs[purpose][img_info["id"]]["filename"] = img_info["file_name"]

                # adjust bbox absolute coordinates for a reshape with padding, and reverse x with y, h with w
                w = img_info["width"]
                h = img_info["height"]

                # TODO
                
    def determine_anchors(self):

        if self.used_categories == {}:
            print("info not yet loaded")
            quit()

        anchor_finder = AnchorFinder(self.imgs)
        self.anchors = anchor_finder.get_anchors()

    def resize_with_pad(self, img):

        w, h = img.shape[0], img.shape[1]

        if w < h:

            if (h - w) % 2 == 0:
                q1 = (h - w) // 2
                q2 = (h - w) // 2
            else:
                q1 = (h - w - 1) // 2
                q2 = (h - w - 1) // 2 + 1

            padl = np.zeros((q1, h, 3), dtype=np.uint8)
            padr = np.zeros((q2, h, 3), dtype=np.uint8)
        
            img = np.concatenate([padl, img, padr], 0)

        elif h < w:

            if (w - h) % 2 == 0:
                q1 = (w - h) // 2
                q2 = (w - h) // 2
            else:
                q1 = (w - h - 1) // 2
                q2 = (w - h - 1) // 2 + 1

            padl = np.zeros((w, q1, 3), dtype=np.uint8)
            padr = np.zeros((w, q2, 3), dtype=np.uint8)

            img = np.concatenate([padl, img, padr], 1)

        return cv.resize(img, self.img_size), self.img_size[0] / max(w, h)

    def assign_anchors_to_objects(self):

        pass