from math import floor
import numpy as np
import tensorflow as tf
import pickle

from utils import *
from metrics import *
from predictions import *
from data_processing import *

class ConvLayer(tf.keras.layers.Layer):

    LEAKY_RELU_RATE = 0.1

    def __init__(self, filters: int, size: int, w_decay: float, stride=1):
        super().__init__()

        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=size, strides=stride, padding="same", kernel_regularizer=tf.keras.regularizers.l2(w_decay))
        self.bnorm = tf.keras.layers.BatchNormalization()
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=ConvLayer.LEAKY_RELU_RATE)

    def call(self, input):
        
        _temp = self.conv(input)
        _temp = self.bnorm(_temp)
        y = self.leaky_relu(_temp)

        return y

class ResBlock(tf.keras.layers.Layer):

    def __init__(self, filters: int, w_decay: float):
        super().__init__()

        self.conv1 = ConvLayer(filters=filters // 2, size=1, w_decay=w_decay)
        self.conv2 = ConvLayer(filters=filters, size=3, w_decay=w_decay)

    def call(self, input):

        _temp = self.conv1(input)
        _temp = self.conv2(_temp)

        # number of output channels is the same as in the input
        # so no conv 1x1
        y = _temp + input

        return y

class ResSequence(tf.keras.layers.Layer):

    def __init__(self, filters: int, res_block_count: int, w_decay: float):
        super().__init__()

        self.intro_conv = ConvLayer(filters=filters, size=3, w_decay=w_decay, stride=2)
        self.res_seq = tf.keras.Sequential([ResBlock(filters=filters, w_decay=w_decay) for _ in range(res_block_count)])

    def call(self, input):
        
        _temp = self.intro_conv(input)
        y = self.res_seq(_temp)

        return y

def build_darknet53_full(CLASS_COUNT):
    
    input_img = tf.keras.layers.Input((IMG_SIZE[0], IMG_SIZE[1], 3))

    conv_back1 = ConvLayer(32, 3, 5e-4)(input_img) # 416
    res_back1 = ResSequence(64, 1, 5e-4)(conv_back1) # 208
    res_back2 = ResSequence(128, 2, 5e-4)(res_back1) # 104
    res_back3 = ResSequence(256, 8, 5e-4)(res_back2) # 52
    res_back4 = ResSequence(512, 8, 5e-4)(res_back3) # 26
    res_back5 = ResSequence(1024, 4, 5e-4)(res_back4) # 13

    # output for scale 1

    features_scale1 = res_back5

    conv_scale1_1 = ConvLayer(512, 1, 5e-4)(features_scale1)
    conv_scale1_2 = ConvLayer(1024, 3, 5e-4)(conv_scale1_1)
    conv_scale1_3 = ConvLayer(512, 1, 5e-4)(conv_scale1_2)
    conv_scale1_4 = ConvLayer(1024, 3, 5e-4)(conv_scale1_3)
    conv_scale1_5 = ConvLayer(512, 1, 5e-4)(conv_scale1_4)

    conv_scale1_6 = ConvLayer(1024, 3, 5e-4)(conv_scale1_5)
    output_scale1 = tf.keras.layers.Conv2D(ANCHOR_PERSCALE_CNT * (4 + 1 + CLASS_COUNT),
                                            1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(5e-4))(conv_scale1_6)

    # output for scale 2

    conv_scale12 = ConvLayer(256, 1, 5e-4)(conv_scale1_5)
    upsample_scale12 = tf.keras.layers.UpSampling2D((2, 2))(conv_scale12)
    features_scale2 = tf.keras.layers.Concatenate(axis=-1)([res_back4, upsample_scale12])

    conv_scale2_1 = ConvLayer(256, 1, 5e-4)(features_scale2)
    conv_scale2_2 = ConvLayer(512, 3, 5e-4)(conv_scale2_1)
    conv_scale2_3 = ConvLayer(256, 1, 5e-4)(conv_scale2_2)
    conv_scale2_4 = ConvLayer(512, 3, 5e-4)(conv_scale2_3)
    conv_scale2_5 = ConvLayer(256, 1, 5e-4)(conv_scale2_4)

    conv_scale2_6 = ConvLayer(512, 3, 5e-4)(conv_scale2_5)
    output_scale2 = tf.keras.layers.Conv2D(ANCHOR_PERSCALE_CNT * (4 + 1 + CLASS_COUNT),
                                            1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(5e-4))(conv_scale2_6)

    # output for scale 3

    conv_scale23 = ConvLayer(128, 1, 5e-4)(conv_scale2_5)
    upsample_scale23 = tf.keras.layers.UpSampling2D((2, 2))(conv_scale23)
    features_scale3 = tf.keras.layers.Concatenate(axis=-1)([res_back3, upsample_scale23])

    conv_scale3_1 = ConvLayer(128, 1, 5e-4)(features_scale3)
    conv_scale3_2 = ConvLayer(256, 3, 5e-4)(conv_scale3_1)
    conv_scale3_3 = ConvLayer(128, 1, 5e-4)(conv_scale3_2)
    conv_scale3_4 = ConvLayer(256, 3, 5e-4)(conv_scale3_3)
    conv_scale3_5 = ConvLayer(128, 1, 5e-4)(conv_scale3_4)

    conv_scale3_6 = ConvLayer(256, 3, 5e-4)(conv_scale3_5)
    output_scale3 = tf.keras.layers.Conv2D(ANCHOR_PERSCALE_CNT * (4 + 1 + CLASS_COUNT),
                                            1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(5e-4))(conv_scale3_6)

    return tf.keras.Model(inputs=input_img, outputs=[output_scale1, output_scale2, output_scale3])

def build_darknet53_encoder(CLASS_COUNT):
    
    pretrain_input_img = tf.keras.layers.Input((PRETRAIN_BOX_SIZE[0], PRETRAIN_BOX_SIZE[1], 3))

    pretrain_conv_back1 = ConvLayer(32, 3, 5e-4)(pretrain_input_img) 
    pretrain_res_back1 = ResSequence(64, 1, 5e-4)(pretrain_conv_back1) 
    pretrain_res_back2 = ResSequence(128, 2, 5e-4)(pretrain_res_back1) 
    pretrain_res_back3 = ResSequence(256, 8, 5e-4)(pretrain_res_back2) 
    pretrain_res_back4 = ResSequence(512, 8, 5e-4)(pretrain_res_back3) 
    pretrain_res_back5 = ResSequence(1024, 4, 5e-4)(pretrain_res_back4)

    pretrain_flatten = tf.keras.layers.Flatten()(pretrain_res_back5)
    pretrain_dense = tf.keras.layers.Dense(CLASS_COUNT)(pretrain_flatten)
    pretrain_classifier = tf.keras.layers.Softmax()(pretrain_dense)

    return tf.keras.Model(inputs=pretrain_input_img, outputs=pretrain_classifier)

def build_mid_full(CLASS_COUNT):
    
    input_img = tf.keras.layers.Input((IMG_SIZE[0], IMG_SIZE[1], 3))

    conv_back1 = ConvLayer(32, 3, 5e-4)(input_img) # 416
    res_back1 = ResSequence(64, 1, 5e-4)(conv_back1) # 208
    res_back2 = ResSequence(128, 2, 5e-4)(res_back1) # 104
    res_back3 = ResSequence(256, 6, 5e-4)(res_back2) # 52
    res_back4 = ResSequence(400, 6, 5e-4)(res_back3) # 26
    res_back5 = ResSequence(800, 4, 5e-4)(res_back4) # 13

    # output for scale 1

    features_scale1 = res_back5

    conv_scale1_1 = ConvLayer(400, 1, 5e-4)(features_scale1)
    conv_scale1_2 = ConvLayer(800, 3, 5e-4)(conv_scale1_1)
    conv_scale1_3 = ConvLayer(400, 1, 5e-4)(conv_scale1_2)
    conv_scale1_4 = ConvLayer(800, 3, 5e-4)(conv_scale1_3)
    conv_scale1_5 = ConvLayer(400, 1, 5e-4)(conv_scale1_4)

    conv_scale1_6 = ConvLayer(800, 3, 5e-4)(conv_scale1_5)
    output_scale1 = tf.keras.layers.Conv2D(ANCHOR_PERSCALE_CNT * (4 + 1 + CLASS_COUNT),
                                            1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(5e-4))(conv_scale1_6)

    # output for scale 2

    conv_scale12 = ConvLayer(256, 1, 5e-4)(conv_scale1_5)
    upsample_scale12 = tf.keras.layers.UpSampling2D((2, 2))(conv_scale12)
    features_scale2 = tf.keras.layers.Concatenate(axis=-1)([res_back4, upsample_scale12])

    conv_scale2_1 = ConvLayer(256, 1, 5e-4)(features_scale2)
    conv_scale2_2 = ConvLayer(400, 3, 5e-4)(conv_scale2_1)
    conv_scale2_3 = ConvLayer(256, 1, 5e-4)(conv_scale2_2)
    conv_scale2_4 = ConvLayer(400, 3, 5e-4)(conv_scale2_3)
    conv_scale2_5 = ConvLayer(256, 1, 5e-4)(conv_scale2_4)

    conv_scale2_6 = ConvLayer(400, 3, 5e-4)(conv_scale2_5)
    output_scale2 = tf.keras.layers.Conv2D(ANCHOR_PERSCALE_CNT * (4 + 1 + CLASS_COUNT),
                                            1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(5e-4))(conv_scale2_6)

    # output for scale 3

    conv_scale23 = ConvLayer(128, 1, 5e-4)(conv_scale2_5)
    upsample_scale23 = tf.keras.layers.UpSampling2D((2, 2))(conv_scale23)
    features_scale3 = tf.keras.layers.Concatenate(axis=-1)([res_back3, upsample_scale23])

    conv_scale3_1 = ConvLayer(128, 1, 5e-4)(features_scale3)
    conv_scale3_2 = ConvLayer(256, 3, 5e-4)(conv_scale3_1)
    conv_scale3_3 = ConvLayer(128, 1, 5e-4)(conv_scale3_2)
    conv_scale3_4 = ConvLayer(256, 3, 5e-4)(conv_scale3_3)
    conv_scale3_5 = ConvLayer(128, 1, 5e-4)(conv_scale3_4)

    conv_scale3_6 = ConvLayer(256, 3, 5e-4)(conv_scale3_5)
    output_scale3 = tf.keras.layers.Conv2D(ANCHOR_PERSCALE_CNT * (4 + 1 + CLASS_COUNT),
                                            1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(5e-4))(conv_scale3_6)

    return tf.keras.Model(inputs=input_img, outputs=[output_scale1, output_scale2, output_scale3])

def build_mid_encoder(CLASS_COUNT):
    
    pretrain_input_img = tf.keras.layers.Input((PRETRAIN_BOX_SIZE[0], PRETRAIN_BOX_SIZE[1], 3))

    pretrain_conv_back1 = ConvLayer(32, 3, 5e-4)(pretrain_input_img) 
    pretrain_res_back1 = ResSequence(64, 1, 5e-4)(pretrain_conv_back1) 
    pretrain_res_back2 = ResSequence(128, 2, 5e-4)(pretrain_res_back1) 
    pretrain_res_back3 = ResSequence(256, 6, 5e-4)(pretrain_res_back2) 
    pretrain_res_back4 = ResSequence(400, 6, 5e-4)(pretrain_res_back3) 
    pretrain_res_back5 = ResSequence(800, 4, 5e-4)(pretrain_res_back4)

    pretrain_flatten = tf.keras.layers.Flatten()(pretrain_res_back5)
    pretrain_dense = tf.keras.layers.Dense(CLASS_COUNT)(pretrain_flatten)
    pretrain_classifier = tf.keras.layers.Softmax()(pretrain_dense)

    return tf.keras.Model(inputs=pretrain_input_img, outputs=pretrain_classifier)

def build_small_full(CLASS_COUNT):
    
    input_img = tf.keras.layers.Input((IMG_SIZE[0], IMG_SIZE[1], 3))

    conv_back1 = ConvLayer(16, 3, 5e-4)(input_img) # 416
    res_back1 = ResSequence(32, 1, 5e-4)(conv_back1) # 208
    res_back2 = ResSequence(64, 2, 5e-4)(res_back1) # 104
    res_back3 = ResSequence(128, 4, 5e-4)(res_back2) # 52
    res_back4 = ResSequence(256, 4, 5e-4)(res_back3) # 26
    res_back5 = ResSequence(512, 2, 5e-4)(res_back4) # 13

    # output for scale 1

    features_scale1 = res_back5

    conv_scale1_1 = ConvLayer(256, 1, 5e-4)(features_scale1)
    conv_scale1_2 = ConvLayer(512, 3, 5e-4)(conv_scale1_1)
    conv_scale1_5 = ConvLayer(256, 1, 5e-4)(conv_scale1_2)

    conv_scale1_6 = ConvLayer(512, 3, 5e-4)(conv_scale1_5)
    output_scale1 = tf.keras.layers.Conv2D(ANCHOR_PERSCALE_CNT * (4 + 1 + CLASS_COUNT),
                                            1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(5e-4))(conv_scale1_6)

    # output for scale 2

    conv_scale12 = ConvLayer(128, 1, 5e-4)(conv_scale1_5)
    upsample_scale12 = tf.keras.layers.UpSampling2D((2, 2))(conv_scale12)
    features_scale2 = tf.keras.layers.Concatenate(axis=-1)([res_back4, upsample_scale12])

    conv_scale2_1 = ConvLayer(128, 1, 5e-4)(features_scale2)
    conv_scale2_2 = ConvLayer(256, 3, 5e-4)(conv_scale2_1)
    conv_scale2_5 = ConvLayer(128, 1, 5e-4)(conv_scale2_2)

    conv_scale2_6 = ConvLayer(256, 3, 5e-4)(conv_scale2_5)
    output_scale2 = tf.keras.layers.Conv2D(ANCHOR_PERSCALE_CNT * (4 + 1 + CLASS_COUNT),
                                            1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(5e-4))(conv_scale2_6)

    # output for scale 3

    conv_scale23 = ConvLayer(64, 1, 5e-4)(conv_scale2_5)
    upsample_scale23 = tf.keras.layers.UpSampling2D((2, 2))(conv_scale23)
    features_scale3 = tf.keras.layers.Concatenate(axis=-1)([res_back3, upsample_scale23])

    conv_scale3_1 = ConvLayer(64, 1, 5e-4)(features_scale3)
    conv_scale3_2 = ConvLayer(128, 3, 5e-4)(conv_scale3_1)
    conv_scale3_5 = ConvLayer(64, 1, 5e-4)(conv_scale3_2)

    conv_scale3_6 = ConvLayer(128, 3, 5e-4)(conv_scale3_5)
    output_scale3 = tf.keras.layers.Conv2D(ANCHOR_PERSCALE_CNT * (4 + 1 + CLASS_COUNT),
                                            1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(5e-4))(conv_scale3_6)

    return tf.keras.Model(inputs=input_img, outputs=[output_scale1, output_scale2, output_scale3])

def build_small_encoder(CLASS_COUNT):
    
    pretrain_input_img = tf.keras.layers.Input((PRETRAIN_BOX_SIZE[0], PRETRAIN_BOX_SIZE[1], 3))

    pretrain_conv_back1 = ConvLayer(16, 3, 5e-4)(pretrain_input_img) 
    pretrain_res_back1 = ResSequence(32, 1, 5e-4)(pretrain_conv_back1) 
    pretrain_res_back2 = ResSequence(64, 2, 5e-4)(pretrain_res_back1) 
    pretrain_res_back3 = ResSequence(128, 4, 5e-4)(pretrain_res_back2) 
    pretrain_res_back4 = ResSequence(256, 4, 5e-4)(pretrain_res_back3) 
    pretrain_res_back5 = ResSequence(512, 2, 5e-4)(pretrain_res_back4)

    pretrain_flatten = tf.keras.layers.Flatten()(pretrain_res_back5)
    pretrain_dense = tf.keras.layers.Dense(CLASS_COUNT)(pretrain_flatten)
    pretrain_classifier = tf.keras.layers.Softmax()(pretrain_dense)

    return tf.keras.Model(inputs=pretrain_input_img, outputs=pretrain_classifier)
