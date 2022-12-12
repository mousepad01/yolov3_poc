from math import floor
import numpy as np
import tensorflow as tf
import pickle

from utils import *
from loss import *
from predictions import *
from data_processing import *

class ConvLayer(tf.keras.layers.Layer):

    LEAKY_RELU_RATE = 0.1

    def __init__(self, filters: int, size: int, w_decay: float, stride=1):
        super().__init__()

        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=size, strides=stride, padding="same", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(w_decay))
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
