from math import floor
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

    LEAKY_RELU_RATE = tf.Constant(0.1)

    def __init__(self, filters: int, size: int, stride=1, padding="valid"):
        super().__init__()

        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=(size, size), strides=(stride, stride), padding=padding)
        self.bnorm = tf.keras.layers.BatchNormalization()
        self.leaky_relu = tf.keras.layers.LeakyReLU(ConvLayer.LEAKY_RELU_RATE)

    def call(self, input):
        
        _temp = self.conv(input)
        _temp = self.bnorm(_temp)
        y = self.leaky_relu(_temp)

        return y

class ResBlock(tf.keras.layers.Layer):

    def __init__(self, filters: int):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=filters // 2, size=(1, 1), stride=(1, 1), padding="same")
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, size=(3, 3), stride=(1, 1), padding="same")

    def call(self, input):

        _temp = self.conv1(input)
        _temp = self.conv2(_temp)

        # number of output channels is the same as in the input
        # so no conv 1x1
        y = _temp + input

        return y

class ResSequence(tf.keras.layers.Layer):

    def __init__(self, filters: int, res_block_count: int):
        super().__init__()

        self.intro_conv = ConvLayer(filters=filters, size=(3, 3), stride=2)
        self.res_seq = tf.keras.Sequential([ResBlock(filters) for _ in range(res_block_count)])

    def call(self, input):
        
        _temp = self.intro_conv(input)
        y = self.res_seq(_temp)

        return y

class Network:

    def __init__(self):

        self.backbone: tf.keras.Model = None
        '''
            backbone for feature extraction (Darknet-53 ???)
        '''

        # self.classification_input_layer: tf.keras.layer = None
        '''
            (UNUSED)
            256 x 256 x 3
        '''

        self.input_layer: tf.keras.layer = None
        '''
            416 x 416 x 3
        '''

        # self.backbone_classification_head: tf.keras.Model = None
        '''
            (UNUSED)
            head for classification task
        '''

        self.full_network: tf.keras.Model = None
        '''
            includes the full network for object detection (so, everything except backbone classification head)
        '''

    def build_components(self, anchors_per_cell=3, class_count=10):
        
        # the backbone
        input_img = tf.keras.layers.Input((IMG_SIZE[0], IMG_SIZE[1], 3))

        conv_back1 = ConvLayer(32, 3)(input_img)
        res_back1 = ResSequence(64, 1)(conv_back1)
        res_back2 = ResSequence(128, 2)(res_back1)
        res_back3 = ResSequence(256, 8)(res_back2)
        res_back4 = ResSequence(512, 8)(res_back3)
        res_back5 = ResSequence(1024, 4)(res_back4)

        self.backbone = tf.keras.Model(inputs=input_img, outputs=res_back5)

        # the entire network

        # output for scale 1

        features_scale1 = res_back5

        conv_scale1_1 = ConvLayer(512, 1)(features_scale1)
        conv_scale1_2 = ConvLayer(1024, 3)(conv_scale1_1)
        conv_scale1_3 = ConvLayer(512, 1)(conv_scale1_2)
        conv_scale1_4 = ConvLayer(1024, 3)(conv_scale1_3)
        conv_scale1_5 = ConvLayer(512, 1)(conv_scale1_4)

        conv_scale1_6 = ConvLayer(1024, 3)(conv_scale1_5)
        output_scale1 = ConvLayer(anchors_per_cell * (4 + 1 + class_count), 1)(conv_scale1_6)

        # output for scale 2

        conv_scale12 = ConvLayer(256, 1)(conv_scale1_5)
        upsample_scale12 = tf.keras.layers.UpSampling2D((2, 2))(conv_scale12)
        features_scale2 = tf.keras.layers.Concatenate(axis=-1)(res_back4, upsample_scale12)

        conv_scale2_1 = ConvLayer(256, 1)(features_scale2)
        conv_scale2_2 = ConvLayer(512, 3)(conv_scale2_1)
        conv_scale2_3 = ConvLayer(256, 1)(conv_scale2_2)
        conv_scale2_4 = ConvLayer(512, 3)(conv_scale2_3)
        conv_scale2_5 = ConvLayer(256, 1)(conv_scale2_4)

        conv_scale2_6 = ConvLayer(512, 3)(conv_scale2_5)
        output_scale2 = ConvLayer(anchors_per_cell * (4 + 1 + class_count), 1)(conv_scale2_6)

        # output for scale 3

        conv_scale23 = ConvLayer(128, 1)(conv_scale1_5)
        upsample_scale23 = tf.keras.layers.UpSampling2D((2, 2))(conv_scale23)
        features_scale3 = tf.keras.layers.Concatenate(axis=-1)(res_back3, upsample_scale23)

        conv_scale3_1 = ConvLayer(256, 1)(features_scale3)
        conv_scale3_2 = ConvLayer(512, 3)(conv_scale3_1)
        conv_scale3_3 = ConvLayer(256, 1)(conv_scale3_2)
        conv_scale3_4 = ConvLayer(512, 3)(conv_scale3_3)
        conv_scale3_5 = ConvLayer(256, 1)(conv_scale3_4)

        conv_scale3_6 = ConvLayer(512, 3)(conv_scale3_5)
        output_scale3 = ConvLayer(anchors_per_cell * (4 + 1 + class_count), 1)(conv_scale3_6)

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
def make_prediction_perscale(output, anchors, THRESHOLD=0.6):
    '''
        output: B x S x S x A x (C + 5)
        anchors: A x 2  --- RELATIVE TO GRID CELL COUNT FOR CURRENT SCALE !!!!
    '''

    S, A = output.shape[1], output.shape[3]

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

    img_px_size = tf.convert_to_tensor(image.shape[:2], dtype=tf.float32)
    img_px_size = tf.reshape(img_px_size, (1, 1, 1, 1, 2))

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

            cv.rectangle(image, (y_min, x_min), (y_max, x_max), color=CLASS_TO_COLOR[predicted_class], thickness=2)
            cv.putText(image, text=f"{class_output}: {floor(predicted_class_p * 100) / 100}%", org=(y_min, x_min - 10), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=CLASS_TO_COLOR[predicted_class], thickness=1)

    cv.imshow("prediction", image)
    cv.waitKey(0)
