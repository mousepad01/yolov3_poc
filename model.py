import numpy as np
import tensorflow as tf
import cv2 as cv

from utils import *

#   TODO:
#       define residual layers used
#       conv - bn - leaky relu blocks
#       loss function with as much preprocessing as possible
#       (in the other file) anchor selection based on k means clustering on training data
#       (?) pretrain on classification tasks
#       (?) train detection on multiple image sizes
#       train loop with different learning rates etc 

class ConvLayer(tf.keras.layers.Layer):

    LEAKY_RELU_RATE = 0.1

    def __init__(self, filters, size, stride=1, padding="valid"):
        super().__init__()

        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=(size, size), strides=(stride, stride), padding=padding)
        self.bnorm = tf.keras.layers.BatchNormalization()
        self.leaky_relu = tf.keras.layers.LeakyReLU(ConvLayer.LEAKY_RELU_RATE)

    def call(self, input):
        
        _temp = self.conv(input)
        _temp = self.bnorm(_temp)
        y = self.leaky_relu(_temp)

        return y

#FIXME
class ResBlock(tf.keras.layers.Layer):

    def __init__(self, filters, size):
        super().__init__()

        self.intro_conv = ConvLayer(filters=filters, size=size, stride=2)
        self.conv1 = tf.keras.layers.Conv2D(filters=filters // 2, size=(1, 1), stride=(1, 1), padding="same")
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, size=(3, 3), stride=(1, 1), padding="same")

    def call(self, input):

        _id = self.intro_conv(input)

        _temp = self.conv1(_id)
        _temp = self.conv2(_temp)

        y = _temp + _id

        return y

# TODO
#@tf.function
def yolov3_loss_persize(output, bool_mask, target_mask):
    '''
        raw output: B x S x S x A x (C + 5)
        (last dimension: tx, ty, tw, th, to, p0, p1, ...p(C-1))

        anchors: A x 2

        bool_mask: B x S x S x A x 1

        target_mask: B x S x S x A x 5
        (last dimension: sigma(tx), sigma(ty), tw, th, i from [0, C - 1])
    '''

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

    c_idx = tf.stack([h_idx, w_idx])
    c_idx = tf.transpose(c_idx)
    c_idx = tf.reshape(c_idx, (1, S, S, 1, 2))

    return c_idx

# FIXME ??? for a single size / multiple
@tf.function
def make_prediction_persize(output, anchors, THRESHOLD=0.6):
    '''
        output: B x S x S x A x C + 5
        anchors: A x 2
    '''

    B, S, A, C = output.shape[0], output.shape[1], output.shape[3], output.shape[4] - 5

    # TODO remove later
    # assert(A == anchors.shape[0])

    anchors = tf.reshape(anchors, (1, 1, 1, A, 2))

    c_idx = get_c_idx(S)
    grid_cells_cnt = tf.convert_to_tensor([S, S]).reshape((1, 1, 1, 1, 2))
    
    # raw
    output_xy = output[..., 0:2]
    output_wh = output[..., 2:4]
    
    # in terms of how many grid cells
    output_xy = tf.sigmoid(output_xy) + c_idx
    output_wh = tf.exp(output_wh) * anchors

    # relative to the whole image
    output_xy = output_xy / grid_cells_cnt
    output_wh = output_wh / grid_cells_cnt

    # corner coordinates
    output_wh_half = output_wh / 2
    output_xy_min = output_xy - output_wh_half
    output_xy_max = output_xy + output_wh_half

    # class probability
    output_class_p_if_object = tf.keras.activations.softmax(output[..., 5:])        # single label classification 
    output_class_p = output_class_p_if_object * tf.sigmoid(output[..., 4])          # confidence gives the probability of being an object

    output_class = tf.argmax(output_class_p, axis=-1)
    output_class_maxp = tf.reduce_max(output_class_p, axis=-1)
    
    output_prediction_mask = output_class_maxp > THRESHOLD
    output_xy_min = tf.boolean_mask(output_xy_min, output_prediction_mask)
    output_xy_max = tf.boolean_mask(output_xy_max, output_prediction_mask)
    output_class = tf.boolean_mask(output_class, output_prediction_mask)
    output_class_maxp = tf.boolean_mask(output_class_maxp, output_prediction_mask)

    return output_xy_min, output_xy_max, output_class, output_class_maxp

# FIXME this is for just one size
def show_predictions(image, pred_xy_min, pred_xy_max, pred_class, pred_class_p):

    img_px_size = tf.convert_to_tensor(image.shape).reshape((1, 1, 1, 1, 2))

    pred_xy_min = pred_xy_min * img_px_size
    pred_xy_max = pred_xy_max * img_px_size
    
    pred_xy_min = tf.reshape(pred_xy_min, (pred_xy_min.shape[0], -1, 2))
    pred_xy_max = tf.reshape(pred_xy_max, (pred_xy_max.shape[0], -1, 2))
    pred_class = tf.reshape(pred_class, (pred_class.shape[0], -1))
    pred_class_p = tf.reshape(pred_class_p, (pred_class_p.shape[0], -1))

    assert(pred_xy_min.shape == pred_xy_max.shape)

    for idx, img in enumerate(image):
        
        for box_idx in range(pred_xy_min[idx].shape[1]):
            
            x_min, y_min = pred_xy_min[idx][box_idx][0], pred_xy_min[idx][box_idx][1]
            x_max, y_max = pred_xy_max[idx][box_idx][0], pred_xy_max[idx][box_idx][1]

            predicted_class = pred_class[idx][box_idx]
            predicted_class_p = pred_class_p[idx][box_idx]

            cv.rectangle(img, (y_min, x_min), (y_max, x_max), color=(0, 0, 255), thickness=2)
            cv.putText(img, text=f"{predicted_class}: {predicted_class_p}%", org=(y_min - 10, x_min), font_face=cv.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255), thickness=2)

        cv.imshow(img)
        if len(image) > 1:
            cv.waitKey(1000)
        else:
            cv.waitKey(0)
