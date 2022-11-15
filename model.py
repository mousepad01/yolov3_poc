import numpy as np
import tensorflow as tf

#   TODO:
#       define residual layers used
#       conv - bn - leaky relu blocks
#       loss function with as much preprocessing as possible
#       (in the other file) anchor selection based on k means clustering on training data
#       (?) pretrain on classification tasks
#       (?) train detection on multiple image sizes
#       train loop with different learning rates etc 

class ConvLayer(tf.keras.layers):

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
class ResBlock(tf.keras.layers):

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

class ResBlockRepeat(tf.keras.layers):
    pass 

def model():

    pass