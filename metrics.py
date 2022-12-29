import tensorflow as tf

from utils import *

@tf.function
def yolov3_loss_perscale(output, obj_mask, ignored_mask, target_mask):
    '''
        raw output: B x S x S x (A * (C + 5))
        (last dimension: tx, ty, tw, th, to, l0, l1, ...l(C-1))

        obj_mask: B x S x S x A x 1

        ignored_mask: B x S x S x A x 1

        target_mask: B x S x S x A x (4 + C)
        (last dimension: tx, ty, tw, th, p0, p1, ...p(C-1))
    '''

    output = tf.reshape(output, (output.shape[0], output.shape[1], output.shape[2], ANCHOR_PERSCALE_CNT, -1))
    CLS_CNT = output.shape[4] - 5

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
    no_obj_mask = 1 - obj_mask - ignored_mask
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
    #output_xy = output[..., 0:2]
    output_wh = output[..., 2:4]
    target_coord_xy = tf.sigmoid(target_mask[..., 0:2])
    #target_coord_xy = target_mask[..., 0:2]
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
