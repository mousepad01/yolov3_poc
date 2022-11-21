import numpy as np
import tensorflow as tf
import cv2 as cv

from data_processing import *
from anchor_kmeans import *
from model import *

def main():
    
    #FIXME
    data_manager = DataManager(train_data_path=DataManager.VALIDATION_DATA_PATH, train_info_path=DataManager.VALIDATION_INFO_PATH)
    data_manager.load_info()
    data_manager.determine_anchors()
    data_manager.assign_anchors_to_objects()

    print("DO NOT FORGET TO ELIMINATE THRESHOLD AT ASSIGN ANCHORS TO OBJECTS")

    print(f"testing loss with {len(data_manager.used_categories.keys())} classes")
    a = tf.random.uniform((16, 13, 13, 3, len(data_manager.used_categories.keys()) + 5))
    yolov3_loss_persize(a, data_manager.bool_anchor_masks[0][:16, ...], data_manager.target_anchor_masks[0][:16, ...])

    # test loss
    #yolov3_loss(None, None)


if __name__ == "__main__":
    main()
