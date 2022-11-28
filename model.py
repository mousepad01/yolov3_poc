from math import floor
import numpy as np
import tensorflow as tf
import cv2 as cv
import pickle

from utils import *
from loss import *
from predictions import *
from data_processing import *

print("FIXME: RE-INTRODUCE BATCH NORM WHEN FINISHING TESTS WITH 1 IMAGE")
print("FIXME: train after done testing code")
print("FIXME: FIX NO OBJ LOSS FACTOR")

class ConvLayer(tf.keras.layers.Layer):

    LEAKY_RELU_RATE = 0.1

    def __init__(self, filters: int, size: int, stride=1):
        super().__init__()

        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=size, strides=stride, padding="same")
        self.bnorm = tf.keras.layers.BatchNormalization()
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=ConvLayer.LEAKY_RELU_RATE)

    def call(self, input):
        
        _temp = self.conv(input)
        #_temp = self.bnorm(_temp)
        y = self.leaky_relu(_temp)

        return y

class ResBlock(tf.keras.layers.Layer):

    def __init__(self, filters: int):
        super().__init__()

        self.conv1 = ConvLayer(filters // 2, 1)
        self.conv2 = ConvLayer(filters, 3)

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

        self.intro_conv = ConvLayer(filters=filters, size=3, stride=2)
        self.res_seq = tf.keras.Sequential([ResBlock(filters) for _ in range(res_block_count)])

    def call(self, input):
        
        _temp = self.intro_conv(input)
        y = self.res_seq(_temp)

        return y

class Network:

    def __init__(self, data_manager):

        self.data_manager: DataManager = data_manager
        '''
            contains data info and all that stuff
        '''

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

    def build_components(self, anchors_per_cell=ANCHOR_PERSCALE_CNT, class_count=10):
        
        # the backbone
        input_img = tf.keras.layers.Input((IMG_SIZE[0], IMG_SIZE[1], 3))

        conv_back1 = ConvLayer(32, 3)(input_img) # 416
        res_back1 = ResSequence(64, 1)(conv_back1) # 208
        res_back2 = ResSequence(128, 2)(res_back1) # 104
        res_back3 = ResSequence(256, 8)(res_back2) # 52
        res_back4 = ResSequence(512, 8)(res_back3) # 26
        res_back5 = ResSequence(1024, 4)(res_back4) # 13

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
        features_scale2 = tf.keras.layers.Concatenate(axis=-1)([res_back4, upsample_scale12])

        conv_scale2_1 = ConvLayer(256, 1)(features_scale2)
        conv_scale2_2 = ConvLayer(512, 3)(conv_scale2_1)
        conv_scale2_3 = ConvLayer(256, 1)(conv_scale2_2)
        conv_scale2_4 = ConvLayer(512, 3)(conv_scale2_3)
        conv_scale2_5 = ConvLayer(256, 1)(conv_scale2_4)

        conv_scale2_6 = ConvLayer(512, 3)(conv_scale2_5)
        output_scale2 = ConvLayer(anchors_per_cell * (4 + 1 + class_count), 1)(conv_scale2_6)

        # output for scale 3

        conv_scale23 = ConvLayer(128, 1)(conv_scale2_5)
        upsample_scale23 = tf.keras.layers.UpSampling2D((2, 2))(conv_scale23)
        features_scale3 = tf.keras.layers.Concatenate(axis=-1)([res_back3, upsample_scale23])

        conv_scale3_1 = ConvLayer(256, 1)(features_scale3)
        conv_scale3_2 = ConvLayer(512, 3)(conv_scale3_1)
        conv_scale3_3 = ConvLayer(256, 1)(conv_scale3_2)
        conv_scale3_4 = ConvLayer(512, 3)(conv_scale3_3)
        conv_scale3_5 = ConvLayer(256, 1)(conv_scale3_4)

        conv_scale3_6 = ConvLayer(512, 3)(conv_scale3_5)
        output_scale3 = ConvLayer(anchors_per_cell * (4 + 1 + class_count), 1)(conv_scale3_6)

        self.full_network = tf.keras.Model(inputs=input_img, outputs=[output_scale1, output_scale2, output_scale3])

    # FIXME
    # TODO use tf.data.Dataset
    def train(self):

        if self.full_network is None:
            print("network not yet initialized")
            quit()

        EPOCHS_STAGE1 = 100
        EPOCHS_STAGE2 = 200
        EPOCHS_STAGE3 = 20

        LR_STAGE1 = 1e-4
        LR_STAGE2 = 1e-5
        LR_STAGE3 = 1e-6

        MOMENTUM = 0.9

        # TODO use this term
        DECAY = 5e-4

        TRAIN_BATCH_SIZE = DATA_LOAD_BATCH_SIZE
        BATCH_CNT = 1 # len(self.data_manager.imgs["train"])

        progbar_output = tf.keras.utils.Progbar(BATCH_CNT)

        # stage 1

        # TODO 
        # first separate training loops for different LR
        # then use lr scheduler

        loss_stats = []
        loss_stats_noobj = []
        loss_stats_obj = []
        loss_stats_cl = []
        loss_stats_xy = []
        loss_stats_wh = []

        epoch_stage = 0
        '''for epochs, optimizer in [(EPOCHS_STAGE1, tf.optimizers.SGD(learning_rate=LR_STAGE1, momentum=MOMENTUM)),
                                    (EPOCHS_STAGE2, tf.optimizers.SGD(learning_rate=LR_STAGE2, momentum=MOMENTUM)),
                                    (EPOCHS_STAGE3, tf.optimizers.SGD(learning_rate=LR_STAGE3, momentum=MOMENTUM))]:'''
        for epochs, optimizer in [(EPOCHS_STAGE1, tf.optimizers.Adam(learning_rate=LR_STAGE1)),
                                    (EPOCHS_STAGE2, tf.optimizers.Adam(learning_rate=LR_STAGE2)),
                                    (EPOCHS_STAGE3, tf.optimizers.Adam(learning_rate=LR_STAGE3))]:

            for epoch in range(epochs):

                tf.print(f"\nEpoch {epoch} (stage {epoch_stage}):")

                sum_loss = 0
                sum_loss_noobj = 0
                sum_loss_obj = 0
                sum_loss_cl = 0
                sum_loss_xy = 0
                sum_loss_wh = 0

                batch_idx = 0
                for (imgs, bool_mask_size1, target_mask_size1, bool_mask_size2, target_mask_size2, bool_mask_size3, target_mask_size3) in self.data_manager.load_train_data(TRAIN_BATCH_SIZE):

                    with tf.GradientTape() as tape:
                    
                        out_s1, out_s2, out_s3 = self.full_network(imgs, training=True)

                        loss_value, noobj, obj, cl, xy, wh = yolov3_loss_perscale(out_s1, bool_mask_size1, target_mask_size1)

                        loss_value_, noobj_, obj_, cl_, xy_, wh_ = yolov3_loss_perscale(out_s2, bool_mask_size2, target_mask_size2)
                        loss_value += loss_value_
                        noobj += noobj_
                        obj += obj_
                        cl += cl_
                        xy += xy_
                        wh += wh_
                        
                        loss_value_, noobj_, obj_, cl_, xy_, wh_ = yolov3_loss_perscale(out_s3, bool_mask_size3, target_mask_size3)
                        loss_value += loss_value_
                        noobj += noobj_
                        obj += obj_
                        cl += cl_
                        xy += xy_
                        wh += wh_

                    #if loss_value > 300:
                     #   break

                    gradients = tape.gradient(loss_value, self.full_network.trainable_weights)
                    optimizer.apply_gradients(zip(gradients, self.full_network.trainable_weights))

                    batch_idx += 1
                    progbar_output.update(batch_idx)
                    sum_loss += loss_value
                    sum_loss_noobj += noobj
                    sum_loss_obj += obj
                    sum_loss_cl += cl
                    sum_loss_xy += xy
                    sum_loss_wh += wh

                    # FIXME
                    break

                #if loss_value > 300:
                 #   break

                tf.print(f"\nLoss value: {floor((sum_loss / BATCH_CNT) * (10 ** LOSS_OUTPUT_PRECISION)) / (10 ** LOSS_OUTPUT_PRECISION)}")
                loss_stats.append(floor((sum_loss / BATCH_CNT) * (10 ** LOSS_OUTPUT_PRECISION)) / (10 ** LOSS_OUTPUT_PRECISION))
                loss_stats_noobj.append(floor((sum_loss_noobj / BATCH_CNT) * (10 ** LOSS_OUTPUT_PRECISION)) / (10 ** LOSS_OUTPUT_PRECISION))
                loss_stats_obj.append(floor((sum_loss_obj / BATCH_CNT) * (10 ** LOSS_OUTPUT_PRECISION)) / (10 ** LOSS_OUTPUT_PRECISION))
                loss_stats_cl.append(floor((sum_loss_cl / BATCH_CNT) * (10 ** LOSS_OUTPUT_PRECISION)) / (10 ** LOSS_OUTPUT_PRECISION))
                loss_stats_xy.append(floor((sum_loss_xy / BATCH_CNT) * (10 ** LOSS_OUTPUT_PRECISION)) / (10 ** LOSS_OUTPUT_PRECISION))
                loss_stats_wh.append(floor((sum_loss_wh / BATCH_CNT) * (10 ** LOSS_OUTPUT_PRECISION)) / (10 ** LOSS_OUTPUT_PRECISION))
                
            epoch_stage += 1

        loss_dump = [loss_stats, loss_stats_noobj, loss_stats_obj, loss_stats_cl, loss_stats_xy, loss_stats_wh]
        with open("loss_stats_dump.bin", "wb+") as fd:
            pickle.dump(loss_dump, fd)

        fig, ax = plt.subplots(3, 2)
        
        ax[0][0].plot([idx for idx in range(len(loss_stats))],
                        loss_stats)
        ax[0][0].grid(True)
        ax[0][0].set_title("total loss")

        ax[0][1].plot([idx for idx in range(len(loss_stats))],
                        loss_stats_noobj)
        ax[0][1].grid(True)
        ax[0][1].set_title("noobj")

        ax[1][0].plot([idx for idx in range(len(loss_stats))],
                        loss_stats_obj)
        ax[1][0].grid(True)
        ax[1][0].set_title("obj")

        ax[1][1].plot([idx for idx in range(len(loss_stats))],
                        loss_stats_cl)
        ax[1][1].grid(True)
        ax[1][1].set_title("cl")

        ax[2][0].plot([idx for idx in range(len(loss_stats))],
                        loss_stats_xy)
        ax[2][0].grid(True)
        ax[2][0].set_title("xy")

        ax[2][1].plot([idx for idx in range(len(loss_stats))],
                        loss_stats_wh)
        ax[2][1].grid(True)
        ax[2][1].set_title("wh")

        plt.show()

    # TODO
    def predict(self):
        
        if self.full_network is None:
            print("network not yet initialized")
            quit()
