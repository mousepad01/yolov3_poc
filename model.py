from math import floor
import numpy as np
import tensorflow as tf
import cv2 as cv

from utils import *
from loss import *
from predictions import *

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

    # TODO
    def train(self):

        if self.full_network is None:
            print("network not yet initialized")
            quit()

        EPOCHS_STAGE1 = 60
        EPOCHS_STAGE2 = 30
        EPOCHS_STAGE3 = 70

        LR_STAGE1 = 1e-3
        LR_STAGE2 = 1e-4
        LR_STAGE3 = 1e-5

        MOMENTUM = 0.9

        # TODO use this term
        DECAY = 5e-4

        # stage 1



        # stage 2

        # stage 3
