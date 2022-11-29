from math import floor
import numpy as np
import tensorflow as tf
import cv2 as cv
import pickle

from utils import *

@tf.function
def _get_c_idx(S):
    '''
        calculate array of shape 1 x S x S x 1 x 2 -> (i, j) in S x S
    '''

    all_idx = tf.range(0, S)

    h_idx = tf.tile(all_idx, (S,))
    
    all_idx = tf.expand_dims(all_idx, 0)
    
    w_idx = tf.tile(all_idx, (S, 1))
    w_idx = tf.transpose(w_idx)
    w_idx = tf.reshape(w_idx, (S * S,))

    c_idx = tf.stack([w_idx, h_idx])
    c_idx = tf.transpose(c_idx)
    c_idx = tf.reshape(c_idx, (1, S, S, 1, 2))

    return c_idx

@tf.function
def make_prediction_perscale(output, anchors, THRESHOLD=0.6):
    '''
        output: B x S x S x (A * (C + 5))
        anchors: A x 2  --- RELATIVE TO GRID CELL COUNT FOR CURRENT SCALE !!!!
    '''

    output = tf.reshape(output, (output.shape[0], output.shape[1], output.shape[2], ANCHOR_PERSCALE_CNT, -1))

    S, A = output.shape[1], output.shape[3]

    # anchors relative to the grid cell count for the current scale
    anchors = tf.cast(tf.reshape(anchors, (1, 1, 1, A, 2)), tf.float32)

    c_idx = _get_c_idx(S)
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
    
    output_prediction_mask = output_class_maxp > THRESHOLD
    output_xy_min = tf.boolean_mask(output_xy_min, output_prediction_mask)
    output_xy_max = tf.boolean_mask(output_xy_max, output_prediction_mask)
    output_class = tf.boolean_mask(output_class, output_prediction_mask)
    output_class_maxp = tf.boolean_mask(output_class_maxp, output_prediction_mask)

    return output_xy_min, output_xy_max, output_class, output_class_maxp

def show_prediction(image, pred_xy_min, pred_xy_max, pred_class, pred_class_p, categ_to_name=None, ground_truth_info=None):
    '''
        pred_xy_min, pred_xy_max: (list of SCALE_CNT=3) 1 x S x S x A x 2
        pred_class: (list of SCALE_CNT=3) 1 x S x S x A x 1
        pred_class_p: (list of SCALE_CNT=3) 1 x S x S x A x 1
        pred_xy are given relative to the whole image size
        (optional) categ_to_name: one hot encoding category idx -> category name
        (optional) ground truth info: [{"category": one hot idx, "bbox": (x, y, w, h) absolute}, ...]
    '''

    image = np.array(image)

    img_px_size = tf.convert_to_tensor(image.shape[:2])
    img_px_size = tf.cast(tf.reshape(img_px_size, (1, 1, 1, 1, 2)), dtype=tf.float32)

    for d in range(SCALE_CNT):

        pred_xy_min[d] = pred_xy_min[d] * img_px_size
        pred_xy_max[d] = pred_xy_max[d] * img_px_size
        
        pred_xy_min[d] = tf.reshape(pred_xy_min[d], (-1, 2))
        pred_xy_max[d] = tf.reshape(pred_xy_max[d], (-1, 2))
        pred_class[d] = tf.reshape(pred_class[d], (-1))
        pred_class_p[d] = tf.reshape(pred_class_p[d], (-1))

    SHOW_RESIZE_FACTOR = 2.3
    image =  cv.resize(image, (int(IMG_SIZE[0] * SHOW_RESIZE_FACTOR), int(IMG_SIZE[0] * SHOW_RESIZE_FACTOR)))

    # if there is ground truth, first show it
    if ground_truth_info is not None:

        image_ = np.copy(image)

        for bbox_d in ground_truth_info:

            predicted_class = int(bbox_d["category"])

            if categ_to_name is not None:
                class_output = categ_to_name[predicted_class]
            else:
                class_output = predicted_class

            x, y, w, h = bbox_d["bbox"]
            x_min, y_min = int(x * SHOW_RESIZE_FACTOR), int(y * SHOW_RESIZE_FACTOR)
            x_max, y_max = int((x + w) * SHOW_RESIZE_FACTOR), int((y + h) * SHOW_RESIZE_FACTOR)

            cv.rectangle(image_, (y_min, x_min), (y_max, x_max), color=CLASS_TO_COLOR[predicted_class], thickness=2)
            cv.putText(image_, text=f"{class_output}", org=(y_min, x_min - 10), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=CLASS_TO_COLOR[predicted_class], thickness=1)

        cv.imshow("ground truth", image_)
        cv.waitKey(0)

    for d in range(SCALE_CNT):
    
        for box_idx in range(pred_xy_min[d].shape[0]):
            
            x_min, y_min = int(pred_xy_min[d][box_idx][0] * SHOW_RESIZE_FACTOR), int(pred_xy_min[d][box_idx][1] * SHOW_RESIZE_FACTOR)
            x_max, y_max = int(pred_xy_max[d][box_idx][0] * SHOW_RESIZE_FACTOR), int(pred_xy_max[d][box_idx][1] * SHOW_RESIZE_FACTOR)

            predicted_class = int(pred_class[d][box_idx])
            predicted_class_p = pred_class_p[d][box_idx]

            if categ_to_name is not None:
                class_output = categ_to_name[predicted_class]
            else:
                class_output = predicted_class

            print(f"prediction: {(y_min, x_min)}, {(y_max, x_max)}, {class_output}: {floor(predicted_class_p * 100)}%")

            cv.rectangle(image, (y_min, x_min), (y_max, x_max), color=CLASS_TO_COLOR[predicted_class], thickness=2)
            cv.putText(image, text=f"{class_output}: {floor(predicted_class_p * 100)}%", org=(y_min, x_min - 10), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=CLASS_TO_COLOR[predicted_class], thickness=1)

    cv.imshow("prediction", image)
    cv.waitKey(0)
