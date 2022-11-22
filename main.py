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

    # test loss function for generic bugs
    #a = tf.random.uniform((16, 13, 13, 3, len(data_manager.used_categories.keys()) + 5))
    #yolov3_loss_persize(a, data_manager.bool_anchor_masks[0][:16, ...], data_manager.target_anchor_masks[0][:16, ...])

    # test target_mask encoding
    # output_from_mask = temp_mask_to_prediction(data_manager.target_anchor_masks, len(data_manager.used_categories.keys()))

    output_from_mask = [None for _ in range(SCALE_CNT)]
    for d in range(SCALE_CNT):

        B, S, A = data_manager.target_anchor_masks[d].shape[0], data_manager.target_anchor_masks[d].shape[1], data_manager.target_anchor_masks[d].shape[3]

        sigmoid_tx_ty = data_manager.target_anchor_masks[d][..., 0:2] * data_manager.bool_anchor_masks[d]
        tw_th = data_manager.target_anchor_masks[d][..., 2:4] * data_manager.bool_anchor_masks[d]
        to = tf.cast(tf.fill((B, S, S, A, 1), value=10.0), dtype=tf.float32) * data_manager.bool_anchor_masks[d] + \
                tf.cast(tf.fill((B, S, S, A, 1), value=-10.0), dtype=tf.float32) * (1 - data_manager.bool_anchor_masks[d])
        probabilities = tf.cast(tf.one_hot(tf.cast(data_manager.target_anchor_masks[d][..., 4], tf.int32), len(data_manager.used_categories.keys())) * 10, dtype=tf.float32)

        output_from_mask[d] = tf.concat([sigmoid_tx_ty, tw_th, to, probabilities], axis=-1) 

    anchors_relative = [tf.cast(GRID_CELL_CNT[d] * (data_manager.anchors[d] / IMG_SIZE[0]), dtype=tf.float32) for d in range(SCALE_CNT)]

    cnt_ = 0
    for img_id in data_manager.imgs["train"].keys():

        cnt_ += 1
        if cnt_ >= 20:
            break

        img = cv.imread(data_manager.data_path["train"] + data_manager.imgs["train"][img_id]["filename"])
        img = data_manager.resize_with_pad(img)

        output_perimg = [make_prediction_perscale(output_from_mask[d][cnt_ - 1: cnt_], anchors_relative[d], 0.6, False) for d in range(SCALE_CNT)]
        show_prediction(img, [output_perimg[d][0] for d in range(SCALE_CNT)],
                                [output_perimg[d][1] for d in range(SCALE_CNT)],
                                [output_perimg[d][2] for d in range(SCALE_CNT)],
                                [output_perimg[d][3] for d in range(SCALE_CNT)],
                                
                        data_manager.onehot_to_name,
                        data_manager.imgs["train"][img_id]["objs"])


if __name__ == "__main__":
    main()
