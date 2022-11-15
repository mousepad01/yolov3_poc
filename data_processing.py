import numpy as np
import tensorflow as tf
import cv2 as cv
import json

class DataManager:
    '''
        NOTE:
            * adapted for object detection with single-label classification (for simplicity)
    '''

    TRAIN_DATA_PATH = "./data/train2017/"
    VALIDATION_DATA_PATH = "./data/val2017/"

    TRAIN_INFO_PATH = "./data/annotations/instances_train2017.json"
    VALIDATION_INFO_PATH = "./data/annotations/instances_val2017.json"

    DATA_LOAD_BATCH_SIZE = 128
    IMG_SIZE = (416, 416)

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

        self.data_load_batch_size = data_load_batch_size

        self.img_size = img_size

    def load_images(self, purpose, reshape_and_tensor=True):
        '''
            generator, for lazy loading
            purpose: "train" | "validation"
            reshape_and_tensor: bool, return in a list or a tensor of reshaped imgs
        '''
        
        if self.used_categories == {}:
            print("info not yet loaded")
            quit()

        current_loaded = []
        for img_id in self.imgs[purpose].keys():

            img = cv.imread(self.data_path[purpose] + self.imgs[purpose][img_id]["filename"])
            if reshape_and_tensor:
                img = cv.resize(img, self.img_size)

            current_loaded.append(img)

            if len(current_loaded) == self.data_load_batch_size:

                if reshape_and_tensor:
                    current_loaded = tf.convert_to_tensor(current_loaded)

                yield current_loaded
                current_loaded = []

        if len(current_loaded) > 0:

            if reshape_and_tensor:
                current_loaded = tf.convert_to_tensor(current_loaded)

            yield current_loaded

    def load_info(self):
        '''
            load everything at once
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
                                                                "w": None,
                                                                "h": None,
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

                self.imgs[purpose][img_info["id"]]["w"] = img_info["width"]
                self.imgs[purpose][img_info["id"]]["h"] = img_info["height"]
                self.imgs[purpose][img_info["id"]]["filename"] = img_info["file_name"]
