import tensorflow as tf

from utils import *

@tf.function
def yolov3_loss_persize(output, bool_mask, target_mask):
    '''
        raw output: B x S x S x (A * (C + 5))
        (last dimension: tx, ty, tw, th, to, p0, p1, ...p(C-1))

        anchors: A x 2

        bool_mask: B x S x S x A x 1

        target_mask: B x S x S x A x 5
        (last dimension: sigma(tx), sigma(ty), tw, th, i from [0, C - 1])
    '''

    output = tf.reshape(output, (output.shape[0], output.shape[1], output.shape[2], ANCHOR_PERSCALE_CNT, -1))

    # sigma(tx), sigma(ty), tw, th     - ti relative to grid cell count for that scale
    output_xy = tf.sigmoid(output[..., 0:2])
    output_wh = output[..., 2:4]
    # TODO try sigma(tx), sigma(ty), e^tw, e^th     - ti relative to grid cell count for that scale

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

    COORD_FACTOR = tf.constant(5.0)
    NOOBJ_FACTOR = tf.constant(.5)

    # FIXME exclude more elements from no obj loss ???
    # no-object loss
    no_object_loss = NOOBJ_FACTOR * (1 - bool_mask) * tf.square(0 - output_confidence)

    # object loss
    object_loss = bool_mask * tf.square(1 - output_confidence)

    # classification loss
    target_class = tf.one_hot(tf.cast(target_mask[..., 4], dtype=tf.int32), output_class_p.shape[4])
    classification_loss = bool_mask * tf.expand_dims(tf.keras.losses.categorical_crossentropy(target_class, output_class_p), axis=-1)

    # coordinates loss
    target_coord_xy = target_mask[..., 0:2]
    target_coord_wh = target_mask[..., 2:4]
    coord_loss = COORD_FACTOR * bool_mask * (tf.square(target_coord_xy - output_xy) + tf.square(target_coord_wh - output_wh))

    total_loss = tf.math.reduce_sum(no_object_loss) + tf.math.reduce_sum(object_loss) + tf.math.reduce_sum(classification_loss) + tf.math.reduce_sum(coord_loss)
    return total_loss

# TODO
@tf.function
def yolov3_loss():
    pass
    