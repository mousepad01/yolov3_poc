import numpy as np
import tensorflow as tf
import cv2 as cv
import random

from data_processing import *
from anchor_kmeans import *
from model import *
from checkpoint_scheds import *
from lr_scheds import *

print("NOTE: this implementation relies on the fact that dictionaries are ORDERED. yielding keys in a nedeterministic order breaks everything")

def main():

    def _test_mask_encoding():

        tf.print("BEFORE START: change dataloader to first YIELD THE KEYS")

        data_loader = DataLoader(cache_key="all")
        data_loader.prepare()

        stats_manager = StatsManager(data_loader.onehot_to_name, iou_thresholds=[], confidence_thresholds=[])

        CLS_CNT = data_loader.get_class_cnt()
        BSIZE = 4

        def _getkeys(l):
            for e in l:
                yield e

        for _ in range(4):

            dl = data_loader.load_data(BSIZE, "train", shuffle=True)
            img_keys = next(dl)
            img_keys = _getkeys(img_keys)

            for (imgs, obj_mask_size1, ignored_mask_size1, target_mask_size1, \
                        obj_mask_size2, ignored_mask_size2, target_mask_size2, \
                        obj_mask_size3, ignored_mask_size3, target_mask_size3, \
                        gt_boxes) in dl:

                img_keys_ = []
                for _ in range(imgs.shape[0]):
                    img_keys_.append(next(img_keys))

                obj_anchor_masks = [obj_mask_size1, obj_mask_size2, obj_mask_size3]
                ignored_anchor_masks = [ignored_mask_size1, ignored_mask_size2, ignored_mask_size3]
                target_anchor_masks = [target_mask_size1, target_mask_size2, target_mask_size3]

                output_from_mask = [None for _ in range(SCALE_CNT)]
                for d in range(SCALE_CNT):

                    B, S, A = target_anchor_masks[d].shape[0], target_anchor_masks[d].shape[1], target_anchor_masks[d].shape[3]
                    tx_ty = target_anchor_masks[d][..., 0:2] * obj_anchor_masks[d]
                    tw_th = target_anchor_masks[d][..., 2:4] * obj_anchor_masks[d]
                    to = tf.cast(tf.fill((B, S, S, A, 1), value=10.0), dtype=tf.float32) * obj_anchor_masks[d] + \
                            tf.cast(tf.fill((B, S, S, A, 1), value=-10.0), dtype=tf.float32) * (1 - obj_anchor_masks[d])
                    probabilities = tf.cast(tf.one_hot(tf.cast(target_anchor_masks[d][..., 4], dtype=tf.int32), CLS_CNT) * 10.0, dtype=tf.float32)
                    output_from_mask[d] = tf.concat([tx_ty, tw_th, to, probabilities], axis=-1) 

                anchors_relative = [tf.cast(GRID_CELL_CNT[d] * (data_loader.anchors[d] / IMG_SIZE[0]), dtype=tf.float32) for d in range(SCALE_CNT)]

                cnt_ = 0
                for img_id in img_keys_:

                    img = np.array(tf.cast(tf.floor(((imgs[cnt_] + 1.0) / 2.0) * 255.0), tf.uint8))
                    
                    print(img_id)
                    cnt_ += 1

                    gt_boxes_ = tf.reshape(gt_boxes[cnt_ - 1: cnt_], (-1, 4))

                    for bb_idx in range(gt_boxes_.shape[0]):

                        gt_b = gt_boxes_[bb_idx]
                        if not (gt_b[0] == 0 and gt_b[1] == 0 and gt_b[2] == 0 and gt_b[3] == 0):
                            
                            xmin = int(gt_b[0] * IMG_SIZE[0] * SHOW_RESIZE_FACTOR)
                            ymin = int(gt_b[1] * IMG_SIZE[0] * SHOW_RESIZE_FACTOR)
                            xmax = int(gt_b[2] * IMG_SIZE[0] * SHOW_RESIZE_FACTOR)
                            ymax = int(gt_b[3] * IMG_SIZE[0] * SHOW_RESIZE_FACTOR)
                            print(f"gt box {(ymin, xmin)}, {(ymax, xmax)}")

                    output = [output_from_mask[d][cnt_ - 1: cnt_] for d in range(SCALE_CNT)]
                    pred_xy_min, pred_xy_max, pred_class, pred_class_p = stats_manager.parse_prediction(output, anchors_relative, 0.95, 100000)

                    stats_manager.show_prediction(img, pred_xy_min, pred_xy_max, pred_class, pred_class_p, data_loader.imgs["train"][img_id]["objs"])

    def _test_loss():

        data_loader = DataLoader(cache_key="all")
        data_loader.prepare()

        BSIZE = 32
        CLS_CNT = data_loader.get_class_cnt()

        for (_, obj_mask_size1, ignored_mask_size1, target_mask_size1, \
                obj_mask_size2, ignored_mask_size2, target_mask_size2, \
                obj_mask_size3, ignored_mask_size3, target_mask_size3) in data_loader.load_data(BSIZE, "train"):

            obj_anchor_masks = [obj_mask_size1, obj_mask_size2, obj_mask_size3]
            ignored_anchor_masks = [ignored_mask_size1, ignored_mask_size2, ignored_mask_size3]
            target_anchor_masks = [target_mask_size1, target_mask_size2, target_mask_size3]

            loss_value, noobj, obj, cl, xy, wh = 0, 0, 0, 0, 0, 0
            for d in range(SCALE_CNT):

                B, S, A = target_anchor_masks[d].shape[0], target_anchor_masks[d].shape[1], target_anchor_masks[d].shape[3]
                t_xywh = target_anchor_masks[d][..., 0:4]
                to = tf.cast(tf.fill((B, S, S, A, 1), value=100.0), dtype=tf.float32) * obj_anchor_masks[d] + \
                        tf.cast(tf.fill((B, S, S, A, 1), value=-100.0), dtype=tf.float32) * (1 - obj_anchor_masks[d] - ignored_anchor_masks[d]) + \
                        tf.random.uniform((B, S, S, A, 1), minval=-10.0, maxval=10.0) * ignored_anchor_masks[d]
                probabilities = tf.cast(tf.one_hot(tf.cast(target_anchor_masks[d][..., 4], dtype=tf.int32), CLS_CNT) * 100.0, dtype=tf.float32)
                output_from_gt = tf.concat([t_xywh, to, probabilities], axis=-1) 

                loss_value_, noobj_, obj_, cl_, xy_, wh_ = yolov3_loss_perscale(output_from_gt, obj_anchor_masks[d], ignored_anchor_masks[d], target_anchor_masks[d])

                loss_value += loss_value_
                noobj += noobj_
                obj += obj_
                cl += cl_
                xy += xy_
                wh += wh_

            print(f"loss {loss_value / BSIZE} (total {loss_value})")
            print(f"noobj {noobj / BSIZE}")
            print(f"obj {obj / BSIZE}")
            print(f"cl {cl / BSIZE}")
            print(f"xy {xy / BSIZE}")
            print(f"wh {wh / BSIZE}")
            print(f"================================\n")

    def _test_boxes():

        data_loader = DataLoader(cache_key="all")
        data_loader.prepare()

        BSIZE = 32

        for (imgs, gt) in data_loader.load_pretrain_data(BSIZE, "train"):

            tf.print("new batch")

            for idx in range(imgs.shape[0]):

                img = np.array(tf.cast(tf.floor(((imgs[idx] + 1.0) / 2.0) * 255.0), tf.uint8))
                #img = np.array(imgs[idx])
                cv.imshow(data_loader.onehot_to_name[int(tf.argmax(gt[idx]))], img)
                cv.waitKey(0)

    def _test_pretrain_baseline():

        BSIZE = 32

        data_loader = DataLoader(cache_key="all")
        data_loader.prepare()

        TRAIN_IMG_CNT = data_loader.get_box_cnt("train")
        VALIDATION_IMG_CNT = data_loader.get_box_cnt("validation")

        def _to_output_t(x): 
            return floor((x / TRAIN_IMG_CNT) * (10 ** LOSS_OUTPUT_PRECISION)) / (10 ** LOSS_OUTPUT_PRECISION)

        def _to_output_v(x):
            return floor((x / VALIDATION_IMG_CNT) * (10 ** LOSS_OUTPUT_PRECISION)) / (10 ** LOSS_OUTPUT_PRECISION)

        CLS_CNT = data_loader.get_class_cnt()

        def _get_rand_pred(b):
            return tf.convert_to_tensor([tf.one_hot(random.randint(0, CLS_CNT - 1), CLS_CNT) for _ in range(b)])

        LR_CH1 = 60
        LR_CH2 = 90
        LR_CH3 = 1000

        lrs = {e: 1e-3 for e in range(LR_CH1)}
        lrs.update({e: 1e-4 for e in range(LR_CH1, LR_CH2)})
        lrs.update({e: 1e-5 for e in range(LR_CH2, LR_CH3)})

        lr_sched = Lr_dict_sched(lrs)
        ch_sched = Minloss_checkpoint([x for x in range(10, 160, 10)])

        model = Network(data_loader, cache_idx="test00000")
        model.build_components(backbone="darknet-53", optimizer=tf.optimizers.SGD(1e-3, momentum=0.9), lr_scheduler=lr_sched, 
                                pretrain_optimizer=tf.optimizers.SGD(1e-3, momentum=0.9), pretrain_lr_scheduler=lr_sched)

        val_loss = 0
        train_loss = 0

        val_acc = 0
        train_acc = 0

        '''for (_, gt) in data_loader.load_pretrain_data(BSIZE, "train"):
            
            out = _get_rand_pred()
            train_loss += encoder_loss(out, gt)
            train_acc += encoder_accuracy(out, gt)'''

        for (_, gt) in data_loader.load_pretrain_data(BSIZE, "validation"):
            
            out = _get_rand_pred(gt.shape[0])
            val_loss += encoder_loss(out, gt)
            val_acc += encoder_accuracy(out, gt)

        val_loss = _to_output_v(val_loss)
        train_loss = _to_output_t(train_loss)
        val_acc = _to_output_v(val_loss)
        train_acc = _to_output_t(train_loss)

        print(val_loss, train_loss, val_acc, train_acc)            

    def _run_training_detonly():

        data_loader = DataLoader(cache_key="all")
        data_loader.prepare()

        LR_CH1 = 60
        LR_CH2 = 90
        LR_CH3 = 200

        lrs = {e: 1e-3 for e in range(LR_CH1)}
        lrs.update({e: 1e-4 for e in range(LR_CH1, LR_CH2)})
        lrs.update({e: 1e-5 for e in range(LR_CH2, LR_CH3)})

        lr_sched = Lr_dict_sched(lrs)
        ch_sched = Minloss_checkpoint([x for x in range(10, 160, 10)])

        model = Network(data_loader, cache_idx="sgd_burnin0")
        model.build_components(backbone="darknet-53", optimizer=tf.optimizers.SGD(1e-3, 0.9), lr_scheduler=lr_sched, 
                                pretrain_optimizer=tf.keras.optimizers.SGD(1e-3, 0.9), pretrain_lr_scheduler=lr_sched)
        #model.pretrain_encoder(10, 32, progbar=True)
        model.train(160, 64, progbar=True, checkpoint_sched=ch_sched, copy_at_checkpoint=False)

    def _run_training():

        data_loader = DataLoader(cache_key="all")
        data_loader.prepare()

        EPOCHS = 160
        P_EPOCHS = 3

        lr_sched = Lr_cosine_decay(1e-6, 1e-5, EPOCHS)

        p_lrs = {e: 1e-3 for e in range(P_EPOCHS)}
        p_lr_sched = Lr_dict_sched(p_lrs)

        ch_sched = Minloss_checkpoint([x for x in range(10, EPOCHS, 1)])

        model = Network(data_loader, cache_idx="test_adam_5e-5_aug3")
        model.copy_model("test_adam_1e-5_aug3")

        model = Network(data_loader, cache_idx="test_adam_1e-5_aug3")
        model.build_components(backbone="darknet-53", optimizer=tf.optimizers.Adam(1e-5), lr_scheduler=lr_sched, 
                                pretrain_optimizer=tf.optimizers.SGD(1e-2, momentum=0.9, nesterov=True), pretrain_lr_scheduler=p_lr_sched)
        model.train(EPOCHS, 32, progbar=False, checkpoint_sched=ch_sched, copy_at_checkpoint=False, save_on_keyboard_interrupt=False)

    def _run_training2():

        data_loader = DataLoader(cache_key="all")
        data_loader.prepare()

        EPOCHS = 160
        P_EPOCHS = 3

        lr_sched = Lr_cosine_decay(5e-6, 5e-5, EPOCHS)

        p_lrs = {e: 1e-3 for e in range(P_EPOCHS)}
        p_lr_sched = Lr_dict_sched(p_lrs)

        ch_sched = Minloss_checkpoint([x for x in range(10, EPOCHS, 1)])

        model = Network(data_loader, cache_idx="test_adam_5e-5_aug3")
        model.build_components(backbone="darknet-53", optimizer=tf.optimizers.Adam(5e-5), lr_scheduler=lr_sched, 
                                pretrain_optimizer=tf.optimizers.SGD(1e-2, momentum=0.9, nesterov=True), pretrain_lr_scheduler=p_lr_sched)
        model.train(EPOCHS, 32, progbar=False, checkpoint_sched=ch_sched, copy_at_checkpoint=False, save_on_keyboard_interrupt=False)

    def _show_stats():

        data_loader = DataLoader(cache_key="all")
        model = Network(data_loader, cache_idx="test_adam_5e-5_aug3")
        #model.plot_pretrain_stats(show_on_screen=True, save_image=False)
        model.plot_train_stats(show_on_screen=True, save_image=False)

    def _test_model():

        data_loader = DataLoader(cache_key="all")
        data_loader.prepare()

        model = Network(data_loader, cache_idx="test_adam_5e-5_aug3")
        model.build_components(backbone="darknet-53", optimizer=tf.optimizers.Adam(5e-5), pretrain_optimizer=tf.optimizers.SGD(1e-3, momentum=0.9))

        model.predict(subset="validation")

    def _find_ap():

        EPOCHS = 160
        P_EPOCHS = 10

        lrs = {e: 5e-5 for e in range(EPOCHS)}
        lr_sched = Lr_dict_sched(lrs)

        p_lrs = {e: 1e-3 for e in range(P_EPOCHS)}
        p_lr_sched = Lr_dict_sched(p_lrs)

        ch_sched = Minloss_checkpoint([x for x in range(10, EPOCHS, 1)])

        data_loader = DataLoader(cache_key="all")
        data_loader.prepare()

        model = Network(data_loader, cache_idx="test_adam_5e-5_aug2")
        model.build_components(backbone="darknet-53", optimizer=tf.optimizers.Adam(5e-5), pretrain_optimizer=tf.optimizers.SGD(1e-3, momentum=0.9))

        model.compute_precision_recall_stats()

        ap_50 = model.get_ap(0.5)
        tf.print(f"\nAP50: {ap_50}")

    #_test_mask_encoding()
    #_test_loss()
    #_test_boxes()
    #_test_for_nan_inf()
    #_test_learning_few_img()
    #_test_pretrain_baseline()
    #_run_training_detonly()
    #_run_training()
    #_run_training2()
    #_show_stats()
    _test_model()
    #_find_ap()

if __name__ == "__main__":
    main()
