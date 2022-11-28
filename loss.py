import tensorflow as tf

from utils import *

#@tf.function
def yolov3_loss_perscale(output, bool_mask, target_mask):
    '''
        raw output: B x S x S x (A * (C + 5))
        (last dimension: tx, ty, tw, th, to, p0, p1, ...p(C-1))

        anchors: A x 2

        bool_mask: B x S x S x A x 1

        target_mask: B x S x S x A x 5
        (last dimension: sigma(tx), sigma(ty), tw, th, i from [0, C - 1])
    '''

    output = tf.reshape(output, (output.shape[0], output.shape[1], output.shape[2], ANCHOR_PERSCALE_CNT, -1))

    # tx, ty, tw, th     - ti relative to grid cell count for that scale
    # output_xy = tf.sigmoid(output[..., 0:2])
    output_xy = output[..., 0:2]
    output_wh = output[..., 2:4]

    # sigma(to)     = Pr(object) (in yolov3) or Pr(object) * IOU(b, object) in yolov2
    output_confidence = tf.sigmoid(output[..., 4:5])

    # softmax over p0, ... p(C-1)
    output_class_p = tf.keras.activations.softmax(output[..., 5:])

    '''
        4 losses:
        * object loss
        * no-object loss
        * classification loss
        * coordinate loss
    '''

    # FIXME exclude more elements from no obj loss ???
    # no-object loss
    neg_bool_mask = 1 - bool_mask
    no_object_loss = neg_bool_mask *  tf.expand_dims(tf.keras.losses.binary_crossentropy(bool_mask, output_confidence), axis=-1)
    no_object_loss = 0.001 * tf.math.reduce_sum(no_object_loss)

    # object loss
    object_loss = bool_mask * tf.expand_dims(tf.keras.losses.binary_crossentropy(bool_mask, output_confidence), axis=-1)
    object_loss = tf.math.reduce_sum(object_loss)

    # classification loss
    target_class = tf.one_hot(tf.cast(target_mask[..., 4], dtype=tf.int32), output_class_p.shape[4])
    classification_loss = bool_mask * tf.expand_dims(tf.keras.losses.categorical_crossentropy(target_class, output_class_p), axis=-1)
    classification_loss = tf.math.reduce_sum(classification_loss)

    # coordinates loss
    target_coord_xy = target_mask[..., 0:2]
    target_coord_wh = target_mask[..., 2:4]
    xy_loss =  bool_mask * tf.square(target_coord_xy - output_xy)
    xy_loss = tf.math.reduce_sum(xy_loss)
    wh_loss =  bool_mask * tf.square(target_coord_wh - output_wh)
    wh_loss = tf.math.reduce_sum(wh_loss)
    coord_loss = xy_loss + wh_loss
    
    print(f"no obj loss = {no_object_loss}")
    print(f"obj loss = {object_loss}")
    print(f"classif loss = {classification_loss}")
    print(f"xy loss = {xy_loss}")
    print(f"wh loss = {wh_loss}")
    print(f"coord loss = {coord_loss}")
    print("\n")
    
    total_loss = no_object_loss + object_loss + classification_loss + coord_loss
    return total_loss, no_object_loss, object_loss, classification_loss, xy_loss, wh_loss
