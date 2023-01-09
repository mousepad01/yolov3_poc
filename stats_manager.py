from math import floor
import numpy as np
import tensorflow as tf
import cv2 as cv

from constants import *
from utils import *

class StatsManager:

    def __init__(self, categ_to_name, iou_thresholds, confidence_thresholds):

        self.categ_to_name = categ_to_name
        '''
            (only for rendering)
            self.categ_to_name[class onehot idx] = class name
        '''

        self.pr_dict = \
            {
                cls:
                {
                    iou_thr: 
                    {
                        obj_thr:
                        {
                            "tp": 0,
                            "tp_fp": 0,
                            "tp_fn": 0,
                        }
                        for obj_thr in confidence_thresholds
                    }
                    for iou_thr in iou_thresholds
                }
                for cls in self.categ_to_name.keys()
            }
        '''
            self.pr_dict[class onehot idx][iou_threshold][obj_threshold] = TP cnt, (TP + FP ~ total predictions) cnt, (TP + FN ~ total gts) cnt
        '''

        self.mAP = None

        self._obj_thrs = confidence_thresholds
        self._iou_thrs = iou_thresholds
        self._cls = list(self.categ_to_name.keys())

    def parse_prediction_perscale(self, output, anchors, obj_threshold):

        output = tf.reshape(output, (output.shape[0], output.shape[1], output.shape[2], ANCHOR_PERSCALE_CNT, -1))

        S, A = output.shape[1], output.shape[3]

        # anchors relative to the grid cell count for the current scale
        anchors = tf.cast(tf.reshape(anchors, (1, 1, 1, A, 2)), tf.float32)

        c_idx = get_c_idx(S)
        grid_cells_cnt = tf.reshape(tf.convert_to_tensor([S, S], dtype=tf.float32), (1, 1, 1, 1, 2))
        
        # raw
        output_xy = output[..., 0:2]
        output_wh = output[..., 2:4]

        # in terms of how many grid cells
        output_xy = tf.sigmoid(output_xy) + tf.cast(c_idx, tf.float32)
        output_wh = tf.exp(output_wh) * anchors 

        # relative to the whole image
        output_xy = output_xy / grid_cells_cnt
        output_wh = output_wh / grid_cells_cnt

        # corner coordinates
        output_wh_half = output_wh / 2
        output_xy_min = output_xy - output_wh_half
        output_xy_max = output_xy + output_wh_half

        # class probability
        output_class_p_if_object = tf.keras.activations.softmax(output[..., 5:])            # single label classification 
        output_class_p = output_class_p_if_object * tf.sigmoid(output[..., 4:5])            # confidence gives the probability of being an object

        output_class = tf.argmax(output_class_p, axis=-1)
        output_class_maxp = tf.reduce_max(output_class_p, axis=-1)
        
        output_prediction_mask = output_class_maxp > obj_threshold
        output_xy_min = tf.boolean_mask(output_xy_min, output_prediction_mask)
        output_xy_max = tf.boolean_mask(output_xy_max, output_prediction_mask)
        output_class = tf.boolean_mask(output_class, output_prediction_mask)
        output_class_maxp = tf.boolean_mask(output_class_maxp, output_prediction_mask)

        return output_xy_min, output_xy_max, output_class, output_class_maxp

    def show_prediction(self, image, pred_xy_min, pred_xy_max, pred_class, pred_class_p, ground_truth_info=None):
        '''
            pred_xy_min, pred_xy_max: PREDS x 2 (absolute coord prediction)
            pred_class: PREDS
            pred_class_p: PREDS
            (optional) ground truth info: [{"category": one hot idx, "bbox": (x, y, w, h) absolute}, ...]
        '''

        image =  cv.resize(image, (int(IMG_SIZE[0] * SHOW_RESIZE_FACTOR), int(IMG_SIZE[1] * SHOW_RESIZE_FACTOR)))

        # if there is ground truth, first show it
        if ground_truth_info is not None:

            image_ = np.copy(image)

            for bbox_d in ground_truth_info:

                predicted_class = int(bbox_d["category"])

                if self.categ_to_name is not None:
                    class_output = self.categ_to_name[predicted_class]
                else:
                    class_output = predicted_class

                x, y, w, h = bbox_d["bbox"]
                x_min, y_min = int(x * SHOW_RESIZE_FACTOR), int(y * SHOW_RESIZE_FACTOR)
                x_max, y_max = int((x + w) * SHOW_RESIZE_FACTOR), int((y + h) * SHOW_RESIZE_FACTOR)

                cv.rectangle(image_, (y_min, x_min), (y_max, x_max), color=CLASS_TO_COLOR[predicted_class], thickness=2)
                cv.putText(image_, text=f"{class_output}", org=(y_min, x_min - 10), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=CLASS_TO_COLOR[predicted_class], thickness=1)

            cv.imshow("ground truth", image_)
            cv.waitKey(0)
        
        for box_idx in range(pred_xy_min.shape[0]):
            
            x_min, y_min = int(pred_xy_min[box_idx][0] * SHOW_RESIZE_FACTOR), int(pred_xy_min[box_idx][1] * SHOW_RESIZE_FACTOR)
            x_max, y_max = int(pred_xy_max[box_idx][0] * SHOW_RESIZE_FACTOR), int(pred_xy_max[box_idx][1] * SHOW_RESIZE_FACTOR)

            predicted_class = int(pred_class[box_idx])
            predicted_class_p = pred_class_p[box_idx]

            if self.categ_to_name is not None:
                class_output = self.categ_to_name[predicted_class]
            else:
                class_output = predicted_class

            tf.print(f"prediction: {(y_min, x_min)}, {(y_max, x_max)}, {class_output}: {floor(predicted_class_p * 100)}%")

            cv.rectangle(image, (y_min, x_min), (y_max, x_max), color=CLASS_TO_COLOR[predicted_class], thickness=2)
            cv.putText(image, text=f"{class_output}: {floor(predicted_class_p * 100)}%", org=(y_min, x_min - 10), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=CLASS_TO_COLOR[predicted_class], thickness=1)

        cv.imshow("prediction", image)
        cv.waitKey(0)

    def parse_prediction(self, output, anchors, obj_threshold=0.6, nms_threshold=0.6):
        '''
            output: (list of SCALE_CNT=3) 1 x S x S x (A * (C + 5))
            anchors: (list of SCALE_CNT=3) A x 2  --- RELATIVE TO GRID CELL COUNT FOR CURRENT SCALE !!!!

            returns (absolute) coordinate predictions, class predictions and objectness predictions:

            pred_xy_min, pred_xy_max: PREDS x 2
            pred_class: PREDS
            pred_class_p: PREDS
        '''

        output_xy_min_scale0, output_xy_max_scale0, output_class_scale0, output_class_maxp_scale0 = self.parse_prediction_perscale(output[0], anchors[0], obj_threshold)
        output_xy_min_scale1, output_xy_max_scale1, output_class_scale1, output_class_maxp_scale1 = self.parse_prediction_perscale(output[1], anchors[1], obj_threshold)
        output_xy_min_scale2, output_xy_max_scale2, output_class_scale2, output_class_maxp_scale2 = self.parse_prediction_perscale(output[2], anchors[2], obj_threshold)

        output_xy_min = [output_xy_min_scale0, output_xy_min_scale1, output_xy_min_scale2]
        output_xy_max = [output_xy_max_scale0, output_xy_max_scale1, output_xy_max_scale2]
        output_class = [output_class_scale0, output_class_scale1, output_class_scale2]
        output_class_maxp = [output_class_maxp_scale0, output_class_maxp_scale1, output_class_maxp_scale2]

        for d in range(SCALE_CNT):

            output_xy_min[d] = output_xy_min[d] * IMG_SIZE[0]
            output_xy_max[d] = output_xy_max[d] * IMG_SIZE[0]

        pred_xy_min, pred_xy_max, pred_class, pred_class_p = non_maximum_supression(output_xy_min, output_xy_max, output_class, output_class_maxp, nms_threshold)
        return pred_xy_min, pred_xy_max, pred_class, pred_class_p

    def update_tp_fp_fn(self, output, gt_info, anchors, nms_threshold):

        gt_xy_min = []
        gt_xy_max = []
        gt_class = []

        for gt_box in gt_info:

            cl = gt_box["category"]
            x, y, w, h = gt_box["bbox"]

            xmin, ymin, xmax, ymax = x, y, x + w, y + h

            gt_xy_min.append(tf.convert_to_tensor([xmin, ymin], dtype=tf.float32))
            gt_xy_max.append(tf.convert_to_tensor([xmax, ymax], dtype=tf.float32))
            gt_class.append(cl)

        for obj_thr in self._obj_thrs:

            pred_xy_min, pred_xy_max, pred_class, _ = self.parse_prediction(output, anchors, obj_thr, nms_threshold)

            gt_cnt = len(gt_xy_min)
            pred_cnt = len(pred_xy_min)

            gt_pred_iou_class = []

            for gt_idx in range(gt_cnt):
                for pred_idx in range(pred_cnt):

                    if gt_class[gt_idx] == pred_class[pred_idx]:

                        iou_value = iou(pred_xy_min[pred_idx], pred_xy_max[pred_idx], gt_xy_min[gt_idx], gt_xy_max[gt_idx])
                        gt_pred_iou_class.append((gt_idx, pred_idx, iou_value, gt_class[gt_idx]))
                        
            gt_pred_iou_class.sort(key=lambda x: x[3], reverse=True)

            for iou_thr in self._iou_thrs:

                # true positives

                used_gt = [False for _ in range(gt_cnt)]
                used_pred = [False for _ in range(pred_cnt)]

                for idx in range(len(gt_pred_iou_class)):
                    gt_idx, pred_idx, iou_value, cl = gt_pred_iou_class[idx]

                    if iou_value > iou_thr and (used_gt[gt_idx] is False) and (used_pred[pred_idx] is False):

                        self.pr_dict[cl][iou_thr][obj_thr]["tp"] += 1

                        used_gt[gt_idx] = True
                        used_pred[pred_idx] = True

                # true positives + false positives

                for pred_idx in range(pred_cnt):
                    self.pr_dict[int(pred_class[pred_idx])][iou_thr][obj_thr]["tp_fp"] += 1

                # true positives + false negatives

                for gt_idx in range(gt_cnt):
                    self.pr_dict[int(gt_class[gt_idx])][iou_thr][obj_thr]["tp_fn"] += 1

    def get_ap(self, iou_threshold):
        '''
            get AP for a given IOU threshold
            (it uses 101-points interpolation)
        '''
        
        pr_rec = \
            {
                cl: []
                for cl in self._cls
            }

        ap = \
            {
                cl: 0
                for cl in self._cls
            }

        for cl in self._cls:

            # calculate precision, recall

            for obj_thr in self._obj_thrs:

                if self.pr_dict[cl][iou_threshold][obj_thr]["tp_fp"] == 0:
                    pr = 0
                else:
                    pr = self.pr_dict[cl][iou_threshold][obj_thr]["tp"] / self.pr_dict[cl][iou_threshold][obj_thr]["tp_fp"]

                if self.pr_dict[cl][iou_threshold][obj_thr]["tp_fn"] == 0:
                    rec = 0
                else:
                    rec = self.pr_dict[cl][iou_threshold][obj_thr]["tp"] / self.pr_dict[cl][iou_threshold][obj_thr]["tp_fn"]

                pr_rec[cl].append((pr, rec))

            pr_rec[cl].sort(key=lambda x: x[1])

            # interpolation and AP

            r_idx = 0
            for inter_rec in range(101):
                ir = inter_rec / 100

                while r_idx < len(pr_rec[cl]) and ir > pr_rec[cl][r_idx][1]:
                    r_idx += 1

                if r_idx < len(pr_rec[cl]):
                    ap[cl] += pr_rec[cl][r_idx][0]

            ap[cl] /= 101

        ap_over_cls = 0
        for ap_ in ap.values():
            ap_over_cls += ap_

        ap_over_cls /= len(self._cls)

        return ap_over_cls

    def get_mean_ap(self):
        '''
            get mAP (AP averaged over all IOU thresholds)
        '''

        mAP = 0
        for iou_thr in self._iou_thrs:
            mAP += self.get_ap(iou_thr)
        mAP /= len(self._iou_thrs)

        return mAP
          