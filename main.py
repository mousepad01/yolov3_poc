import numpy as np
import tensorflow as tf
import cv2 as cv

from data_processing import *
from anchor_kmeans import *
from model import *

print("NOTE: this implementation relies on the fact that dictionaries are ORDERED. yielding keys in a nedeterministic order breaks everything")

def tests():

    def _test_mask_encoding():

        data_manager = DataManager(train_data_path=DataManager.VALIDATION_DATA_PATH, train_info_path=DataManager.VALIDATION_INFO_PATH)
        data_manager.load_info()
        data_manager.determine_anchors()
        data_manager.assign_anchors_to_objects()

        output_from_mask = [None for _ in range(SCALE_CNT)]
        for d in range(SCALE_CNT):

            B, S, A = data_manager.target_anchor_masks[d].shape[0], data_manager.target_anchor_masks[d].shape[1], data_manager.target_anchor_masks[d].shape[3]
            #sigmoid_tx_ty = data_manager.target_anchor_masks[d][..., 0:2]
            #tx_ty = tf.math.log(sigmoid_tx_ty / (1 - sigmoid_tx_ty)) * data_manager.bool_anchor_masks[d]
            tx_ty = data_manager.target_anchor_masks[d][..., 0:2] * data_manager.bool_anchor_masks[d]
            tw_th = data_manager.target_anchor_masks[d][..., 2:4] * data_manager.bool_anchor_masks[d]
            to = tf.cast(tf.fill((B, S, S, A, 1), value=10.0), dtype=tf.float32) * data_manager.bool_anchor_masks[d] + \
                    tf.cast(tf.fill((B, S, S, A, 1), value=-10.0), dtype=tf.float32) * (1 - data_manager.bool_anchor_masks[d])
            probabilities = data_manager.target_anchor_masks[d][..., 4:] * 10
            output_from_mask[d] = tf.concat([tx_ty, tw_th, to, probabilities], axis=-1) 

        anchors_relative = [tf.cast(GRID_CELL_CNT[d] * (data_manager.anchors[d] / IMG_SIZE[0]), dtype=tf.float32) for d in range(SCALE_CNT)]

        cnt_ = 0
        for img_id in data_manager.imgs["train"].keys():
            
            print(img_id)
            cnt_ += 1

            img = cv.imread(data_manager.data_path["train"] + data_manager.imgs["train"][img_id]["filename"])
            img = data_manager.resize_with_pad(img)

            output_perimg = [make_prediction_perscale(output_from_mask[d][cnt_ - 1: cnt_], anchors_relative[d], 0.6) for d in range(SCALE_CNT)]
            show_prediction(img, [output_perimg[d][0] for d in range(SCALE_CNT)],
                                    [output_perimg[d][1] for d in range(SCALE_CNT)],
                                    [output_perimg[d][2] for d in range(SCALE_CNT)],
                                    [output_perimg[d][3] for d in range(SCALE_CNT)],
                                    
                            data_manager.onehot_to_name,
                            data_manager.imgs["train"][img_id]["objs"])

    def _test_learning_one_img():

        # BEFORE RUNNING: 
        # make sure training takes place only on the first img

        #FIXME
        data_manager = DataManager(train_data_path=DataManager.VALIDATION_DATA_PATH, train_info_path=DataManager.VALIDATION_INFO_PATH)
        data_manager.load_info()
        data_manager.determine_anchors()
        data_manager.assign_anchors_to_objects()

        model = Network(data_manager)
        model.build_components(backbone="small")

        model.train()

        for (img, bool_mask_size1, target_mask_size1, bool_mask_size2, target_mask_size2, bool_mask_size3, target_mask_size3) in data_manager.load_train_data(1):

            out_scale1, out_scale2, out_scale3 = model.full_network(img)

            loss_value, noobj, obj, cl, xy, wh = yolov3_loss_perscale(out_scale1, bool_mask_size1, target_mask_size1)
            print(loss_value, noobj, obj, cl, xy, wh)
            loss_value, noobj, obj, cl, xy, wh = yolov3_loss_perscale(out_scale2, bool_mask_size2, target_mask_size2)
            print(loss_value, noobj, obj, cl, xy, wh)
            loss_value, noobj, obj, cl, xy, wh = yolov3_loss_perscale(out_scale3, bool_mask_size3, target_mask_size3)
            print(loss_value, noobj, obj, cl, xy, wh)

            anchors_relative = [tf.cast(GRID_CELL_CNT[d] * (data_manager.anchors[d] / IMG_SIZE[0]), dtype=tf.float32) for d in range(SCALE_CNT)]
        
            output_xy_min_scale0, output_xy_max_scale0, output_class_scale0, output_class_maxp_scale0 = make_prediction_perscale(out_scale1, anchors_relative[0], 0.6)
            output_xy_min_scale1, output_xy_max_scale1, output_class_scale1, output_class_maxp_scale1 = make_prediction_perscale(out_scale2, anchors_relative[1], 0.6)
            output_xy_min_scale2, output_xy_max_scale2, output_class_scale2, output_class_maxp_scale2 = make_prediction_perscale(out_scale3, anchors_relative[2], 0.6)

            output_xy_min = [output_xy_min_scale0, output_xy_min_scale1, output_xy_min_scale2]
            output_xy_max = [output_xy_max_scale0, output_xy_max_scale1, output_xy_max_scale2]
            output_class = [output_class_scale0, output_class_scale1, output_class_scale2]
            output_class_maxp = [output_class_maxp_scale0, output_class_maxp_scale1, output_class_maxp_scale2]

            show_prediction(np.array(img[0]), output_xy_min, output_xy_max, output_class, output_class_maxp, data_manager.onehot_to_name)

            break

    def _plot_model_stats():

        data_manager = DataManager(train_data_path=DataManager.VALIDATION_DATA_PATH, train_info_path=DataManager.VALIDATION_INFO_PATH)
        data_manager.load_info()
        data_manager.determine_anchors()
        data_manager.assign_anchors_to_objects()

        model = Network(data_manager)
        model.build_components()

        model.show_stats()

    def _test_cache():

        data_manager = DataManager(train_data_path=DataManager.VALIDATION_DATA_PATH, train_info_path=DataManager.VALIDATION_INFO_PATH, cache_key="0")
        data_manager.load_info()
        data_manager.determine_anchors()
        data_manager.assign_anchors_to_objects()

        print(data_manager.target_anchor_masks[0].shape, data_manager.target_anchor_masks[1].shape, data_manager.target_anchor_masks[2].shape)

    def _test_train():

        # BEFORE RUNNING: 
        # make sure training takes place only on the first img

        data_manager = DataManager(train_data_path=DataManager.VALIDATION_DATA_PATH, train_info_path=DataManager.VALIDATION_INFO_PATH)
        data_manager.load_info()
        data_manager.determine_anchors()
        data_manager.assign_anchors_to_objects()

        model = Network(data_manager)
        model.build_components(backbone="small")

        model.train()

        for (img, bool_mask_size1, target_mask_size1, bool_mask_size2, target_mask_size2, bool_mask_size3, target_mask_size3) in data_manager.load_train_data(1):

            out_scale1, out_scale2, out_scale3 = model.full_network(img)

            anchors_relative = [tf.cast(GRID_CELL_CNT[d] * (data_manager.anchors[d] / IMG_SIZE[0]), dtype=tf.float32) for d in range(SCALE_CNT)]
        
            output_xy_min_scale0, output_xy_max_scale0, output_class_scale0, output_class_maxp_scale0 = make_prediction_perscale(out_scale1, anchors_relative[0], 0.6)
            output_xy_min_scale1, output_xy_max_scale1, output_class_scale1, output_class_maxp_scale1 = make_prediction_perscale(out_scale2, anchors_relative[1], 0.6)
            output_xy_min_scale2, output_xy_max_scale2, output_class_scale2, output_class_maxp_scale2 = make_prediction_perscale(out_scale3, anchors_relative[2], 0.6)

            output_xy_min = [output_xy_min_scale0, output_xy_min_scale1, output_xy_min_scale2]
            output_xy_max = [output_xy_max_scale0, output_xy_max_scale1, output_xy_max_scale2]
            output_class = [output_class_scale0, output_class_scale1, output_class_scale2]
            output_class_maxp = [output_class_maxp_scale0, output_class_maxp_scale1, output_class_maxp_scale2]

            show_prediction(np.array(img[0]), output_xy_min, output_xy_max, output_class, output_class_maxp, data_manager.onehot_to_name)

            break

    #_test_mask_encoding()
    #_test_learning_one_img()
    #_plot_model_stats()
    _test_cache()
    #_test_train()
    
def main():
    
    tests()
    
if __name__ == "__main__":
    main()
