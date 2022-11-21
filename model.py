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
@tf.function
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

@tf.function
def make_prediction_perscale(output, anchors, THRESHOLD=0.6):
    '''
        output: B x S x S x A x (C + 5)
        anchors: A x 2  --- RELATIVE TO GRID CELL COUNT FOR CURRENT SCALE !!!!
    '''

    S, A = output.shape[1], output.shape[3]

    # TODO remove later
    # assert(A == anchors.shape[0])

    # anchors relative to the grid cell count for the current scale
    anchors = tf.reshape(anchors, (1, 1, 1, A, 2))

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

    print(output_xy_min.shape, output_xy_max.shape)

    # class probability
    output_class_p_if_object = tf.keras.activations.softmax(output[..., 5:])            # single label classification 
    output_class_p = output_class_p_if_object * tf.sigmoid(output[..., 4:5])            # confidence gives the probability of being an object

    output_class = tf.argmax(output_class_p, axis=-1)
    output_class_maxp = tf.reduce_max(output_class_p, axis=-1)

    print(output_class.shape, output_class_maxp.shape, output_class_p.shape)
    print(output_class_maxp)
    
    output_prediction_mask = output_class_maxp > THRESHOLD
    print(output_prediction_mask.shape)

    output_xy_min = tf.boolean_mask(output_xy_min, output_prediction_mask)
    output_xy_max = tf.boolean_mask(output_xy_max, output_prediction_mask)
    output_class = tf.boolean_mask(output_class, output_prediction_mask)
    output_class_maxp = tf.boolean_mask(output_class_maxp, output_prediction_mask)

    print(output_xy_min.shape, output_xy_max.shape)
    print(output_class.shape, output_class_p.shape)

    return output_xy_min, output_xy_max, output_class, output_class_maxp

# test function to check the results of target_mask encoding from assign_anchors_to_objects()
def temp_mask_to_prediction(masks_per_scale, C):
    '''
        masks_per_scale: (list of 3=SCALE_CNT) B x S x S x A x 5
        C: number of classes
    '''

    output = [None for _ in range(SCALE_CNT)]
    for d in range(SCALE_CNT):

        B, S, A = masks_per_scale[d][0], masks_per_scale[d][1], masks_per_scale[d][3]

        output[d] = tf.concat([tf.math.log(masks_per_scale[d][..., 0:2] / (1 - masks_per_scale[d][..., 0:2])), 
                                masks_per_scale[d][..., 2:4],
                                tf.fill((B, S, S, A, 1), value=10),
                                tf.one_hot(masks_per_scale[d][..., 4], C)
                                ])
        print(output[d].shape)

    return output

def show_prediction(image, pred_xy_min, pred_xy_max, pred_class, pred_class_p):
    '''
        pred_xy_min, pred_xy_max: (list of SCALE_CNT=3) 1 x S x S x A x 2
        pred_class: (list of SCALE_CNT=3) 1 x S x S x A x 1
        pred_class_p: (list of SCALE_CNT=3) 1 x S x S x A x 1
        pred_xy are given relative to the whole image size
    '''

    print(pred_xy_min[0].shape, pred_xy_max[0].shape, pred_class[0].shape, pred_class_p[0].shape)

    img_px_size = tf.convert_to_tensor(image.shape[:2], dtype=tf.float32)
    img_px_size = tf.reshape(img_px_size, (1, 1, 1, 1, 2))

    for d in range(SCALE_CNT):

        pred_xy_min[d] = pred_xy_min[d] * img_px_size
        pred_xy_max[d] = pred_xy_max[d] * img_px_size
        
        pred_xy_min[d] = tf.reshape(pred_xy_min[d], (pred_xy_min[d].shape[0], -1, 2))
        pred_xy_max[d] = tf.reshape(pred_xy_max[d], (pred_xy_max[d].shape[0], -1, 2))
        pred_class[d] = tf.reshape(pred_class[d], (pred_class[d].shape[0], -1))
        pred_class_p[d] = tf.reshape(pred_class_p[d], (pred_class_p[d].shape[0], -1))

        assert(pred_xy_min[d].shape == pred_xy_max[d].shape)

    for d in range(SCALE_CNT):
    
        for box_idx in range(pred_xy_min[d].shape[1]):
            
            x_min, y_min = pred_xy_min[d][box_idx][0], pred_xy_min[d][box_idx][1]
            x_max, y_max = pred_xy_max[d][box_idx][0], pred_xy_max[d][box_idx][1]

            predicted_class = pred_class[d][box_idx]
            predicted_class_p = pred_class_p[d][box_idx]

            cv.rectangle(image, (y_min, x_min), (y_max, x_max), color=(0, 0, 255), thickness=2)
            cv.putText(image, text=f"{predicted_class}: {predicted_class_p}%", org=(y_min - 10, x_min), font_face=cv.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255), thickness=2)

    cv.imshow("prediction", image)
    cv.waitKey(0)
