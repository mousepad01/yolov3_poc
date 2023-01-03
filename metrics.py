import tensorflow as tf

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
def _iou(output_xy_min, output_xy_max, gt_xy_min, gt_xy_max):
    
    lo = tf.maximum(output_xy_min, gt_xy_min)
    hi = tf.minimum(output_xy_max, gt_xy_max)

    difs = tf.maximum(hi - lo, 0)

    intersection = difs[..., 0] * difs[..., 1]
    
    output_difs = output_xy_max - output_xy_min
    gt_difs = gt_xy_max - gt_xy_min

    union = output_difs[..., 0] * output_difs[..., 1] + gt_difs[..., 0] * gt_difs[..., 1] - intersection

    return intersection / union

@tf.function
def yolov3_loss_perscale(output, obj_mask, ignored_mask, target_mask, anchors, gt_boxes):
    '''
        raw output: B x S x S x (A * (C + 5))
        (last dimension: tx, ty, tw, th, to, l0, l1, ...l(C-1))

        obj_mask: B x S x S x A x 1

        ignored_mask: B x S x S x A x 1

        target_mask: B x S x S x A x (4 + C)
        (last dimension: tx, ty, tw, th, p0, p1, ...p(C-1))

        anchors: 1 x 1 x 1 x A x 2

        gt_boxes: 1 x 1 x 1 x 1 x GT x 4
    '''

    output = tf.reshape(output, (output.shape[0], output.shape[1], output.shape[2], ANCHOR_PERSCALE_CNT, -1))
    CLS_CNT = output.shape[4] - 5
    S = output.shape[1]

    # check which objects to dynamically ignore from no-objectness

    c_idx = _get_c_idx(S)
    grid_cells_cnt = tf.reshape(tf.convert_to_tensor([S, S], dtype=tf.float32), (1, 1, 1, 1, 2))
    
    output_xy = output[..., 0:2]
    output_wh = output[..., 2:4]

    output_xy = tf.sigmoid(output_xy) + tf.cast(c_idx, tf.float32)
    output_wh = tf.exp(output_wh) * anchors 

    output_xy = output_xy / grid_cells_cnt
    output_wh = output_wh / grid_cells_cnt

    output_wh_half = output_wh / 2
    output_xy_min = output_xy - output_wh_half
    output_xy_max = output_xy + output_wh_half

    output_xy_min = tf.expand_dims(output_xy_min, axis=4)
    output_xy_max = tf.expand_dims(output_xy_max, axis=4)

    gt_xy_min = gt_boxes[..., 0:2]
    gt_xy_max = gt_boxes[..., 2:4]

    pred_gt_iou = _iou(output_xy_min, output_xy_max, gt_xy_min, gt_xy_max)
    # B x S x S x A x GT

    max_pred_gt_iou = tf.reduce_max(pred_gt_iou, axis=4)
    max_pred_gt_iou = tf.expand_dims(max_pred_gt_iou, axis=-1)
    dyn_ignored = max_pred_gt_iou > IGNORED_IOU_THRESHOLD

    ignored_final_mask = tf.cast(ignored_mask, tf.bool)
    ignored_final_mask = tf.math.logical_or(ignored_final_mask, dyn_ignored)
    ignored_final_mask = tf.cast(ignored_final_mask, tf.float32)

    '''
        4 losses:
        * object loss
        * no-object loss
        * classification loss
        * coordinate loss
    '''

    NO_OBJ_COEF = tf.constant(0.5)
    OBJ_COEFF = tf.constant(1.0)
    CLASSIF_COEFF = tf.constant(1.0)
    XY_COEFF = tf.constant(5.0)
    WH_COEFF = tf.constant(5.0)

    # no-object loss
    output_confidence = tf.sigmoid(output[..., 4:5])
    no_obj_mask = (1 - obj_mask) * (1 - ignored_final_mask)
    no_object_loss = no_obj_mask *  tf.expand_dims(tf.keras.losses.binary_crossentropy(obj_mask, output_confidence), axis=-1)
    no_object_loss = NO_OBJ_COEF * tf.math.reduce_sum(no_object_loss)

    # object loss
    object_loss = obj_mask * tf.expand_dims(tf.keras.losses.binary_crossentropy(obj_mask, output_confidence), axis=-1)
    object_loss = OBJ_COEFF * tf.math.reduce_sum(object_loss)

    # classification loss
    output_class_p = tf.keras.activations.softmax(output[..., 5:])
    target_class = target_mask[..., 4]
    target_class = tf.one_hot(tf.cast(target_class, dtype=tf.int32), CLS_CNT)
    classification_loss = obj_mask * tf.expand_dims(tf.keras.losses.categorical_crossentropy(target_class, output_class_p), axis=-1)
    classification_loss = CLASSIF_COEFF * tf.math.reduce_sum(classification_loss)

    # coordinates loss
    output_xy = tf.sigmoid(output[..., 0:2])
    target_coord_xy = tf.sigmoid(target_mask[..., 0:2])
    #output_xy = output[..., 0:2]
    #target_coord_xy = target_mask[..., 0:2]
    output_wh = output[..., 2:4]
    target_coord_wh = target_mask[..., 2:4]
    xy_loss =  obj_mask * tf.square(target_coord_xy - output_xy)
    xy_loss = XY_COEFF * tf.math.reduce_sum(xy_loss)
    wh_loss =  obj_mask * tf.square(target_coord_wh - output_wh)
    wh_loss = WH_COEFF * tf.math.reduce_sum(wh_loss)
    coord_loss = (xy_loss + wh_loss)
    
    total_loss = no_object_loss + object_loss + classification_loss + coord_loss
    return total_loss, no_object_loss, object_loss, classification_loss, xy_loss, wh_loss

@tf.function
def encoder_loss(output, gt):
    return tf.math.reduce_sum(tf.keras.losses.categorical_crossentropy(gt, output))

@tf.function
def encoder_accuracy(output, gt):
    return tf.reduce_sum(tf.cast(tf.equal(tf.argmax(output, axis=-1), tf.argmax(gt, axis=-1)), dtype=tf.int32))
