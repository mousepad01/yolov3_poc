import numpy as np
import tensorflow as tf
import cv2 as cv

from data_processing import *
from anchor_kmeans import *
from model import *
from custom import *

print("NOTE: this implementation relies on the fact that dictionaries are ORDERED. yielding keys in a nedeterministic order breaks everything")

def main():

    def _test_mask_encoding():

        data_loader = DataLoader(cache_key="zebra_bottle_keyboard", classes=["zebra", "bottle", "keyboard"], superclasses=[])
        data_loader.load_info()
        data_loader.determine_anchors()
        data_loader.assign_anchors_to_objects()

        # hack to get img ids
        def _get_imgid():
            for imgid in data_loader.imgs["validation"].keys():
                yield imgid

        for _ in range(1):

            img_keys = _get_imgid()
            for (imgs, bool_mask_size1, target_mask_size1, bool_mask_size2, target_mask_size2, bool_mask_size3, target_mask_size3) in data_loader.load_data(4, "validation"):

                img_keys_ = []
                for _ in range(imgs.shape[0]):
                    img_keys_.append(next(img_keys))

                bool_anchor_masks = [bool_mask_size1, bool_mask_size2, bool_mask_size3]
                target_anchor_masks = [target_mask_size1, target_mask_size2, target_mask_size3]

                output_from_mask = [None for _ in range(SCALE_CNT)]
                for d in range(SCALE_CNT):

                    B, S, A = target_anchor_masks[d].shape[0], target_anchor_masks[d].shape[1], target_anchor_masks[d].shape[3]
                    tx_ty = target_anchor_masks[d][..., 0:2] * bool_anchor_masks[d]
                    tw_th = target_anchor_masks[d][..., 2:4] * bool_anchor_masks[d]
                    to = tf.cast(tf.fill((B, S, S, A, 1), value=10.0), dtype=tf.float32) * bool_anchor_masks[d] + \
                            tf.cast(tf.fill((B, S, S, A, 1), value=-10.0), dtype=tf.float32) * (1 - bool_anchor_masks[d])
                    probabilities = target_anchor_masks[d][..., 4:] * 10
                    output_from_mask[d] = tf.concat([tx_ty, tw_th, to, probabilities], axis=-1) 

                anchors_relative = [tf.cast(GRID_CELL_CNT[d] * (data_loader.anchors[d] / IMG_SIZE[0]), dtype=tf.float32) for d in range(SCALE_CNT)]

                cnt_ = 0
                for img_id in img_keys_:
                    
                    print(img_id)
                    cnt_ += 1

                    img = cv.imread(data_loader.data_path["validation"] + data_loader.imgs["validation"][img_id]["filename"])
                    img = data_loader.cache_manager.resize_with_pad(img)

                    output_perimg = [make_prediction_perscale(output_from_mask[d][cnt_ - 1: cnt_], anchors_relative[d], 0.6) for d in range(SCALE_CNT)]
                    show_prediction(img, [output_perimg[d][0] for d in range(SCALE_CNT)],
                                            [output_perimg[d][1] for d in range(SCALE_CNT)],
                                            [output_perimg[d][2] for d in range(SCALE_CNT)],
                                            [output_perimg[d][3] for d in range(SCALE_CNT)],
                                            
                                    data_loader.onehot_to_name,
                                    data_loader.imgs["validation"][img_id]["objs"])

    def _test_learning_one_img():

        # BEFORE RUNNING: 
        # make sure training takes place only on the first img

        data_loader = DataLoader(cache_key="mouse_keyboard_tv_laptop", classes=["mouse", "keyboard", "tv", "laptop"], superclasses=[])
        data_loader.load_info()
        data_loader.determine_anchors()
        data_loader.assign_anchors_to_objects()

        def _lr_sched(epoch, lr):

            if epoch < 800:
                return 1e-4

            elif epoch < 1600:
                return 1e-5

            else:
                return 1e-6

        model = Network(data_loader, cache_idx="overfit1")
        model.build_components(backbone="small", optimizer=tf.optimizers.Adam(learning_rate=1e-4), lr_scheduler=_lr_sched)
        
        model.train(1715, 1)

        for (img, bool_mask_size1, target_mask_size1, bool_mask_size2, target_mask_size2, bool_mask_size3, target_mask_size3) in data_loader.load_data(1, "train"):

            out_scale1, out_scale2, out_scale3 = model.full_network(img)

            loss_value, noobj, obj, cl, xy, wh = yolov3_loss_perscale(out_scale1, bool_mask_size1, target_mask_size1)
            print(loss_value, noobj, obj, cl, xy, wh)
            loss_value, noobj, obj, cl, xy, wh = yolov3_loss_perscale(out_scale2, bool_mask_size2, target_mask_size2)
            print(loss_value, noobj, obj, cl, xy, wh)
            loss_value, noobj, obj, cl, xy, wh = yolov3_loss_perscale(out_scale3, bool_mask_size3, target_mask_size3)
            print(loss_value, noobj, obj, cl, xy, wh)

            anchors_relative = [tf.cast(GRID_CELL_CNT[d] * (data_loader.anchors[d] / IMG_SIZE[0]), dtype=tf.float32) for d in range(SCALE_CNT)]
        
            output_xy_min_scale0, output_xy_max_scale0, output_class_scale0, output_class_maxp_scale0 = make_prediction_perscale(out_scale1, anchors_relative[0], 0.6)
            output_xy_min_scale1, output_xy_max_scale1, output_class_scale1, output_class_maxp_scale1 = make_prediction_perscale(out_scale2, anchors_relative[1], 0.6)
            output_xy_min_scale2, output_xy_max_scale2, output_class_scale2, output_class_maxp_scale2 = make_prediction_perscale(out_scale3, anchors_relative[2], 0.6)

            output_xy_min = [output_xy_min_scale0, output_xy_min_scale1, output_xy_min_scale2]
            output_xy_max = [output_xy_max_scale0, output_xy_max_scale1, output_xy_max_scale2]
            output_class = [output_class_scale0, output_class_scale1, output_class_scale2]
            output_class_maxp = [output_class_maxp_scale0, output_class_maxp_scale1, output_class_maxp_scale2]

            show_prediction(np.array(img[0]), output_xy_min, output_xy_max, output_class, output_class_maxp, data_loader.onehot_to_name)

            break

    def _plot_model_stats():

        data_loader = DataLoader(train_data_path=DataLoader.VALIDATION_DATA_PATH, train_info_path=DataLoader.VALIDATION_INFO_PATH)
        data_loader.load_info()
        data_loader.determine_anchors()
        data_loader.assign_anchors_to_objects()

        model = Network(data_loader)
        model.build_components()

        model.show_architecture_stats()

    def _test_cache():

        data_loader = DataLoader(cache_key="base")
        data_loader.load_info()
        data_loader.determine_anchors()
        data_loader.assign_anchors_to_objects()

    def _test_learning_few_img():

        FEW = 2

        data_loader = DataLoader(cache_key="base")
        data_loader.load_info()
        data_loader.determine_anchors()
        data_loader.assign_anchors_to_objects()

        def _lr_sched(epoch, lr):

            if epoch < 800:
                return 1e-4

            elif epoch < 1600:
                return 1e-5

            else:
                return 1e-6

        model = Network(data_loader, cache_idx="overfit_on2_1")
        model.build_components(backbone="small", optimizer=tf.optimizers.Adam(learning_rate=1e-4), lr_scheduler=_lr_sched)
        
        model.train(1700, FEW)

        for (img, bool_mask_size1, target_mask_size1, bool_mask_size2, target_mask_size2, bool_mask_size3, target_mask_size3) in data_loader.load_data(FEW, "train"):
            for idx in range(img.shape[0]):

                out_scale1, out_scale2, out_scale3 = model.full_network(img[idx: idx + 1])

                loss_value, noobj, obj, cl, xy, wh = yolov3_loss_perscale(out_scale1, bool_mask_size1[idx: idx + 1], target_mask_size1[idx: idx + 1])
                print(loss_value, noobj, obj, cl, xy, wh)
                loss_value, noobj, obj, cl, xy, wh = yolov3_loss_perscale(out_scale2, bool_mask_size2[idx: idx + 1], target_mask_size2[idx: idx + 1])
                print(loss_value, noobj, obj, cl, xy, wh)
                loss_value, noobj, obj, cl, xy, wh = yolov3_loss_perscale(out_scale3, bool_mask_size3[idx: idx + 1], target_mask_size3[idx: idx + 1])
                print(loss_value, noobj, obj, cl, xy, wh)

                anchors_relative = [tf.cast(GRID_CELL_CNT[d] * (data_loader.anchors[d] / IMG_SIZE[0]), dtype=tf.float32) for d in range(SCALE_CNT)]
            
                output_xy_min_scale0, output_xy_max_scale0, output_class_scale0, output_class_maxp_scale0 = make_prediction_perscale(out_scale1, anchors_relative[0], 0.6)
                output_xy_min_scale1, output_xy_max_scale1, output_class_scale1, output_class_maxp_scale1 = make_prediction_perscale(out_scale2, anchors_relative[1], 0.6)
                output_xy_min_scale2, output_xy_max_scale2, output_class_scale2, output_class_maxp_scale2 = make_prediction_perscale(out_scale3, anchors_relative[2], 0.6)

                output_xy_min = [output_xy_min_scale0, output_xy_min_scale1, output_xy_min_scale2]
                output_xy_max = [output_xy_max_scale0, output_xy_max_scale1, output_xy_max_scale2]
                output_class = [output_class_scale0, output_class_scale1, output_class_scale2]
                output_class_maxp = [output_class_maxp_scale0, output_class_maxp_scale1, output_class_maxp_scale2]

                show_prediction(np.array(img[idx]), output_xy_min, output_xy_max, output_class, output_class_maxp, data_loader.onehot_to_name)

            break

    def _run_training():

        data_loader = DataLoader(cache_key="zebra_bottle_keyboard2", classes=["zebra", "bottle", "keyboard"], superclasses=[])
        #data_loader = DataLoader(cache_key="base2")
        data_loader.load_info()
        data_loader.determine_anchors()
        data_loader.assign_anchors_to_objects()

        def _lr_sched(epoch, lr):

            if epoch < 60:
                return 1e-3

            elif epoch < 90:
                return 1e-4

            else:
                return 1e-5

        model = Network(data_loader, cache_idx="friday_night")
        model.build_components(backbone="small", optimizer=tf.optimizers.Adam(1e-3), lr_scheduler=_lr_sched)

        def _checkpoint_sched(epoch, loss, vloss):

            if epoch % 5 == 0:
                return True

            return False

        model.train(220, 32, _checkpoint_sched)
        #model.plot_train_stats(show_on_screen=True, save_image=False)

    def _show_stats():

        data_loader = DataLoader(cache_key="zebra_bottle_keyboard2", classes=["zebra", "bottle", "keyboard"], superclasses=[])
        model = Network(data_loader, cache_idx="friday_night")
        model.plot_stats(show_on_screen=True, save_image=False)

    #_test_mask_encoding()
    #_test_learning_one_img()
    #_plot_model_stats()
    #_test_cache()
    #_test_train()
    #_test_learning_few_img()
    #_run_training()
    _show_stats()
    
if __name__ == "__main__":
    main()
