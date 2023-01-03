import tensorflow as tf

from constants import *

@tf.function
def get_c_idx(S):
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
def iou(fst_xy_min, fst_xy_max, snd_xy_min, snd_xy_max):
    '''
        vectorized IOU
    '''
    
    lo = tf.maximum(fst_xy_min, snd_xy_min)
    hi = tf.minimum(fst_xy_max, snd_xy_max)

    difs = tf.maximum(hi - lo, 0)

    intersection = difs[..., 0] * difs[..., 1]
    
    fst_difs = fst_xy_max - fst_xy_min
    snd_difs = snd_xy_max - snd_xy_min

    union = fst_difs[..., 0] * fst_difs[..., 1] + snd_difs[..., 0] * snd_difs[..., 1] - intersection

    return intersection / union

def non_maximum_supression(pred_xy_min, pred_xy_max, pred_class, pred_class_p, iou_threshold):
    '''
        pred_xy_min, pred_xy_max: (list of SCALE_CNT=3) PREDS x 2
        pred_class: (list of SCALE_CNT=3) PREDS
        pred_class_p: (list of SCALE_CNT=3) PREDS

        returns tensors with the same shapes, but w/o scale axis, and with <= element count
    '''

    predictions = [(pred_xy_min[d][idx], pred_xy_max[d][idx], pred_class[d][idx], pred_class_p[d][idx]) for d in range(SCALE_CNT) for idx in range(pred_xy_min[d].shape[0])]
    predictions.sort(key=lambda x: x[3], reverse=True)

    nms_xy_min = []
    nms_xy_max = []
    nms_class = []
    nms_class_p = []

    while len(predictions) > 0:

        best_pred = predictions.pop(0)

        idx = 0
        while idx < len(predictions):
            
            if iou(best_pred[0], best_pred[1], predictions[idx][0], predictions[idx][1]) >= iou_threshold:
                predictions.pop(idx)

            else:
                idx += 1

        nms_xy_min.append(best_pred[0])
        nms_xy_max.append(best_pred[1])
        nms_class.append(best_pred[2])
        nms_class_p.append(best_pred[3])

    nms_xy_min = tf.convert_to_tensor(nms_xy_min)
    nms_xy_max = tf.convert_to_tensor(nms_xy_max)
    nms_class = tf.convert_to_tensor(nms_class)
    nms_class_p = tf.convert_to_tensor(nms_class_p)

    return nms_xy_min, nms_xy_max, nms_class, nms_class_p
