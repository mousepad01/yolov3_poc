from math import floor
import numpy as np
import tensorflow as tf
import pickle
import json

from custom_keras import *
from checkpoint_scheds import *
from lr_scheds import *
from constants import *
from metrics import *
from stats_manager import *
from data_processing import *

class Network:

    '''
        Network states
    '''

    NOT_CREATED = 0
    UNTRAINED = 1
    TRAINING_ENCODER = 2
    TRAINING_DETECTION = 3

    def __init__(self, data_loader, cache_idx=None):

        self._status = Network.NOT_CREATED
        '''
            internal status
        '''

        self.data_loader: DataLoader = data_loader
        '''
            contains data info and all that stuff
        '''

        self.next_train_epoch = 0
        '''
            the next epoch index to be executed when training (the full network)
            (0 if the model is new)
        '''

        self.next_pretrain_epoch = 0
        '''
            the next epoch index to be executed when pr-training (the encoder)
            (0 if the model is new)
        '''

        self.cache_manager = NetworkCacheManager(self, self.data_loader.cache_manager.cache_key, cache_idx)
        '''
            the cache manager
        '''

        self.stats_manager, self.stats_computed = self.cache_manager.get_stats_manager()
        '''
            it is responsible for prediction rendering and model stats other than the loss (mAP, PR curve)
        '''
        
        self.encoder: tf.keras.Model = None
        '''
            the encoder (common backbone with the full network + a classifier head, and different input)
        '''

        self.full_network: tf.keras.Model = None
        '''
            full network for object detection 
        '''

        self.lr_scheduler: function = None
        '''
            schedule the learning rate during training
            * NOTE: this function is not saved when the training is interrupred; it is best this function is defined stateless
        '''

        self.pretrain_lr_scheduler: function = None
        '''
            schedule the learning rate during pre-training
            * NOTE: this function is not saved when the training is interrupred; it is best this function is defined stateless
        '''

    def copy_model(self, new_cache_idx):
        '''
            Copy model cache and its stats under new cache_idx (same cache_key)
        '''

        if self._status is Network.NOT_CREATED:
            tf.print("Model has not been created. Nothing to copy.")
            return

        self.cache_manager.copy_model(new_cache_idx)
       
    def _update_encoder_weights(self):
        '''
            internal method for transferring encoder weights 
            from the encoder to the full network
        '''

        '''
            NOTE: the classification head has 3 layers, and the first layer is the input,
                    so the range is defined as follows
        '''
        for idx in range(1, len(self.encoder.layers) - 3):
            
            ws = self.encoder.get_layer(index=idx).get_weights()
            self.full_network.get_layer(index=idx).set_weights(ws)

    def build_components(self, optimizer: tf.keras.optimizers.Optimizer, pretrain_optimizer: tf.keras.optimizers.Optimizer, \
                                lr_scheduler=lambda epoch, lr: lr, pretrain_lr_scheduler=lambda epoch, lr: lr, \
                                backbone="darknet-53"):
        ''' 
            if there is already a saved model and cache is used, it loads that model
            if not, it builds the model  using the given parameters
            * optimizer: to use during training (ignored if a saved model is loaded)
            * lr_scheduler(current epoch, current lr) => updated lr
            * backbone: string with the name of the backbone to use (ignored if a saved model is loaded)
            * weight_decay: for all convolutions
        '''

        if self._status > Network.NOT_CREATED:
            tf.print("Model has already been created.")
            return

        self.lr_scheduler = lr_scheduler
        self.pretrain_lr_scheduler = pretrain_lr_scheduler

        self.cache_manager.get_model()
        if self.full_network is not None:
            return

        tf.print("Building a new model...")

        CLASS_COUNT = self.data_loader.get_class_cnt()

        if backbone == "darknet-53":
        
            self.full_network = build_darknet53_full(CLASS_COUNT)
            self.encoder = build_darknet53_encoder(CLASS_COUNT)

        elif backbone == "mid":

            self.full_network = build_mid_full(CLASS_COUNT)
            self.encoder = build_mid_encoder(CLASS_COUNT)

        elif backbone == "small":

            self.full_network = build_small_full(CLASS_COUNT)
            self.encoder = build_small_encoder(CLASS_COUNT)

        else:
            print("unknown backbone")
            quit()
        
        self.full_network.compile(optimizer=optimizer)
        self.encoder.compile(optimizer=pretrain_optimizer)

        self._status = Network.UNTRAINED

    def plot_train_stats(self, show_on_screen=False, save_image=True):
        '''
            loads AND shows the train statistics under the cache_key, cache_idx entry
            * show_on_screen: if True, show on screen
            * save_image: if True, save as image in the TRAIN_STATS_PATH
        '''

        train_loss_stats, train_loss_stats_noobj, train_loss_stats_obj, \
        train_loss_stats_cl, train_loss_stats_xy, train_loss_stats_wh, \
        validation_loss_stats, validation_loss_stats_noobj, validation_loss_stats_obj, \
        validation_loss_stats_cl, validation_loss_stats_xy, validation_loss_stats_wh = self.cache_manager.get_train_stats()

        _, ax = plt.subplots(3, 2)

        ax[0][0].plot([idx for idx in range(len(train_loss_stats))],
                        train_loss_stats, label="train", color='blue')
        ax[0][0].plot([idx for idx in range(len(validation_loss_stats))],
                        validation_loss_stats, label="val", color='green')
        ax[0][0].grid(True)
        ax[0][0].set_title("total loss")
        ax[0][0].legend()

        ax[0][1].plot([idx for idx in range(len(train_loss_stats))],
                        train_loss_stats_noobj, label="train", color='blue')
        ax[0][1].plot([idx for idx in range(len(validation_loss_stats))],
                        validation_loss_stats_noobj, label="val", color='green')
        ax[0][1].grid(True)
        ax[0][1].set_title("(no-)objectness loss")
        ax[0][1].legend()

        ax[1][0].plot([idx for idx in range(len(train_loss_stats))],
                        train_loss_stats_obj, label="train", color='blue')
        ax[1][0].plot([idx for idx in range(len(validation_loss_stats))],
                        validation_loss_stats_obj, label="val", color='green')
        ax[1][0].grid(True)
        ax[1][0].set_title("objectness loss")
        ax[1][0].legend()

        ax[1][1].plot([idx for idx in range(len(train_loss_stats))],
                        train_loss_stats_cl, label="train", color='blue')
        ax[1][1].plot([idx for idx in range(len(validation_loss_stats))],
                        validation_loss_stats_cl, label="val", color='green')
        ax[1][1].grid(True)
        ax[1][1].set_title("classification loss")
        ax[1][1].legend()

        ax[2][0].plot([idx for idx in range(len(train_loss_stats))],
                        train_loss_stats_xy, label="train", color='blue')
        ax[2][0].plot([idx for idx in range(len(validation_loss_stats))],
                        validation_loss_stats_xy, label="val", color='green')
        ax[2][0].grid(True)
        ax[2][0].set_title("x-y loss")
        ax[2][0].legend()

        ax[2][1].plot([idx for idx in range(len(train_loss_stats))],
                        train_loss_stats_wh, label="train", color='blue')
        ax[2][1].plot([idx for idx in range(len(validation_loss_stats))],
                        validation_loss_stats_wh, label="val", color='green')
        ax[2][1].grid(True)
        ax[2][1].set_title("w-h loss")
        ax[2][1].legend()

        if save_image:
            self.cache_manager.store_train_stats_fig()

        if show_on_screen:
            plt.show()
      
    def plot_pretrain_stats(self, show_on_screen=False, save_image=True):
        '''
            loads AND shows the pre-train statistics under the cache_key, cache_idx entry
            * show_on_screen: if True, show on screen
            * save_image: if True, save as image in the TRAIN_STATS_PATH
        '''

        train_loss_stats, validation_loss_stats, train_accuracy_stats, validation_accuracy_stats = self.cache_manager.get_pretrain_stat()

        _, ax = plt.subplots(1, 2)

        ax[0][0].plot([idx for idx in range(len(train_loss_stats))],
                        train_loss_stats, label="train", color='blue')
        ax[0][0].plot([idx for idx in range(len(validation_loss_stats))],
                        validation_loss_stats, label="val", color='green')
        ax[0][0].grid(True)
        ax[0][0].set_title("Loss")
        ax[0][0].legend()

        ax[0][1].plot([idx for idx in range(len(train_loss_stats))],
                        train_accuracy_stats, label="train", color='blue')
        ax[0][1].plot([idx for idx in range(len(validation_loss_stats))],
                        validation_accuracy_stats, label="val", color='green')
        ax[0][1].grid(True)
        ax[0][1].set_title("Accuracy")
        ax[0][1].legend()

        if save_image:
            self.cache_manager.store_pretrain_stats_fig()

        if show_on_screen:
            plt.show()

    def pretrain_encoder(self, epochs, batch_size, progbar=True, checkpoint_sched=lambda epoch, loss, vloss: False, copy_at_checkpoint=True):
        '''
            * epochs: number of total epochs (effective number of epochs executed: epochs - self.next_train_epoch + 1)
            * batch_size: for training
            * checkpoint_sched: decide whether to create a checkpoint, after the end of an epoch
        '''

        if self._status is Network.NOT_CREATED:
            tf.print("Network not yet initialized")
            return

        if self._status > Network.TRAINING_ENCODER:
            tf.print("Cannot train encoder after detection training has already started.")
            return

        self._status = Network.TRAINING_ENCODER

        TRAIN_BATCH_SIZE = batch_size
        VALIDATION_BATCH_SIZE = batch_size

        TRAIN_IMG_CNT = self.data_loader.get_box_cnt("train")
        VALIDATION_IMG_CNT = self.data_loader.get_box_cnt("validation")

        TRAIN_BATCH_CNT = TRAIN_IMG_CNT // TRAIN_BATCH_SIZE
        if TRAIN_IMG_CNT % TRAIN_BATCH_SIZE > 0:
            TRAIN_BATCH_CNT += 1

        VALIDATION_BATCH_CNT = VALIDATION_IMG_CNT // VALIDATION_BATCH_SIZE
        if VALIDATION_IMG_CNT % VALIDATION_BATCH_SIZE > 0:
            VALIDATION_BATCH_CNT += 1

        train_loss_stats, validation_loss_stats, train_accuracy_stats, validation_accuracy_stats = self.cache_manager.get_pretrain_stat()

        def _to_output_t(x): 
            return floor((x / TRAIN_IMG_CNT) * (10 ** LOSS_OUTPUT_PRECISION)) / (10 ** LOSS_OUTPUT_PRECISION)

        def _to_output_v(x):
            return floor((x / VALIDATION_IMG_CNT) * (10 ** LOSS_OUTPUT_PRECISION)) / (10 ** LOSS_OUTPUT_PRECISION)

        def _log_show_losses():

            train_accuracy_stats.append(_to_output_t(sum_acc))
            train_loss_stats.append(_to_output_t(sum_loss))

            validation_accuracy_stats.append(_to_output_v(val_acc))
            validation_loss_stats.append(_to_output_v(val_loss))

            #tf.print(f"\n===================================================================================================================\n")
            tf.print(f"\nTrain loss:            {_to_output_t(sum_loss)}")
            tf.print(f"\nTrain accuracy:        {_to_output_t(sum_acc)}")
            tf.print(f"\n")
            tf.print(f"\nValidation loss:       {_to_output_v(val_loss)}")
            tf.print(f"\nValidation accuracy:   {_to_output_v(val_acc)}")
            tf.print(f"\n===================================================================================================================\n")

        for epoch in range(self.next_pretrain_epoch, epochs, 1):

            try:

                if progbar:
                    progbar_output = tf.keras.utils.Progbar(TRAIN_BATCH_CNT)
                tf.print(f"\n(Pretrain) Epoch {epoch}:")

                # loss stats variables

                sum_loss = 0
                sum_acc = 0

                val_loss = 0
                val_acc = 0

                # train loop

                batch_idx = 0
                for (imgs, gt) in self.data_loader.load_pretrain_data(TRAIN_BATCH_SIZE, "train", shuffle=True, augment_probability=0.7):

                    new_lr = self.pretrain_lr_scheduler(epoch, batch_idx, self.encoder.optimizer.learning_rate)
                    self.encoder.optimizer.learning_rate = new_lr

                    with tf.GradientTape() as tape:
                        
                        out = self.encoder(imgs, training=True)
                        loss_value = encoder_loss(out, gt)
                        acc_value = encoder_accuracy(out, gt)

                    gradients = tape.gradient(loss_value, self.encoder.trainable_weights)
                    self.encoder.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_weights))

                    batch_idx += 1
                    if progbar:
                        progbar_output.update(batch_idx)

                    sum_loss += loss_value
                    sum_acc += acc_value

                    #break

                #continue

                # validation loop

                for (imgs, gt) in self.data_loader.load_pretrain_data(VALIDATION_BATCH_SIZE, "validation", augment_probability=0):
                    
                    out = self.encoder(imgs, training=False)
                    val_loss += encoder_loss(out, gt)
                    val_acc += encoder_accuracy(out, gt)

            except KeyboardInterrupt:

                tf.print(f"\nPre-training paused at epoch {epoch}")

                self.next_pretrain_epoch = epoch
                self._update_encoder_weights()
                self.cache_manager.store_model()

                stats = [train_loss_stats, validation_loss_stats, train_accuracy_stats, validation_accuracy_stats]
                self.cache_manager.store_pretrain_stats(stats)

                return

            _log_show_losses()

            if checkpoint_sched(epoch, sum_loss, val_loss):
                
                self.next_pretrain_epoch = epoch + 1
                self._update_encoder_weights()
                self.cache_manager.store_model()

                stats = [train_loss_stats, validation_loss_stats, train_accuracy_stats, validation_accuracy_stats]
                self.cache_manager.store_pretrain_stats(stats)

                if copy_at_checkpoint:
                    vloss = _to_output_v(val_loss)
                    self.cache_manager.copy_model(f"{self.cache_manager.cache_idx}_e{epoch}_vloss{vloss}")

        if self.next_pretrain_epoch >= epochs:
            tf.print(f"\nModel is already pre-trained (at least) {epochs} epochs.")

        else:
            tf.print(f"\nPre-raining is done ({epochs} epochs).")

            self.next_pretrain_epoch = epochs
            self._update_encoder_weights()
            self.cache_manager.store_model()

            stats = [train_loss_stats, validation_loss_stats, train_accuracy_stats, validation_accuracy_stats]
            self.cache_manager.store_pretrain_stats(stats)

    def train(self, epochs, batch_size, progbar=True, checkpoint_sched=lambda epoch, loss, vloss: False, copy_at_checkpoint=True, 
                save_on_keyboard_interrupt=True):
        '''
            * epochs: number of total epochs (effective number of epochs executed: epochs - self.next_train_epoch + 1)
            * batch_size: for training
            * checkpoint_sched: decide whether to create a checkpoint, after the end of an epoch
        '''

        if self._status is Network.NOT_CREATED:
            tf.print("Network not yet initialized")
            return

        self._status = Network.TRAINING_DETECTION

        anchors_relative = [tf.cast(GRID_CELL_CNT[d] * (self.data_loader.anchors[d] / IMG_SIZE[0]), dtype=tf.float32) for d in range(SCALE_CNT)]

        TRAIN_BATCH_SIZE = batch_size
        VALIDATION_BATCH_SIZE = batch_size

        TRAIN_IMG_CNT = self.data_loader.get_img_cnt("train")
        VALIDATION_IMG_CNT = self.data_loader.get_img_cnt("validation")

        TRAIN_BATCH_CNT = TRAIN_IMG_CNT // TRAIN_BATCH_SIZE
        if TRAIN_IMG_CNT % TRAIN_BATCH_SIZE > 0:
            TRAIN_BATCH_CNT += 1

        VALIDATION_BATCH_CNT = VALIDATION_IMG_CNT // VALIDATION_BATCH_SIZE
        if VALIDATION_IMG_CNT % VALIDATION_BATCH_SIZE > 0:
            VALIDATION_BATCH_CNT += 1

        train_loss_stats, train_loss_stats_noobj, train_loss_stats_obj, \
        train_loss_stats_cl, train_loss_stats_xy, train_loss_stats_wh, \
        validation_loss_stats, validation_loss_stats_noobj, validation_loss_stats_obj, \
        validation_loss_stats_cl, validation_loss_stats_xy, validation_loss_stats_wh = self.cache_manager.get_train_stats()

        def _to_output_t(x): 
            return floor((x / TRAIN_IMG_CNT) * (10 ** LOSS_OUTPUT_PRECISION)) / (10 ** LOSS_OUTPUT_PRECISION)

        def _to_output_v(x):
            return floor((x / VALIDATION_IMG_CNT) * (10 ** LOSS_OUTPUT_PRECISION)) / (10 ** LOSS_OUTPUT_PRECISION)

        def _log_show_losses():

            train_loss_stats_noobj.append(_to_output_t(sum_loss_noobj))
            train_loss_stats_obj.append(_to_output_t(sum_loss_obj))
            train_loss_stats_cl.append(_to_output_t(sum_loss_cl))
            train_loss_stats_xy.append(_to_output_t(sum_loss_xy))
            train_loss_stats_wh.append(_to_output_t(sum_loss_wh))
            train_loss_stats.append(_to_output_t(sum_loss))

            validation_loss_stats_noobj.append(_to_output_v(val_loss_noobj))
            validation_loss_stats_obj.append(_to_output_v(val_loss_obj))
            validation_loss_stats_cl.append(_to_output_v(val_loss_cl))
            validation_loss_stats_xy.append(_to_output_v(val_loss_xy))
            validation_loss_stats_wh.append(_to_output_v(val_loss_wh))
            validation_loss_stats.append(_to_output_v(val_loss))

            #tf.print(f"\n===================================================================================================================\n")
            tf.print(f"\nTrain total loss:           {_to_output_t(sum_loss)}")
            tf.print(f"\nTrain (no-)objectness loss: {_to_output_t(sum_loss_noobj)}")
            tf.print(f"\nTrain objectness loss:      {_to_output_t(sum_loss_obj)}")
            tf.print(f"\nTrain classification loss:  {_to_output_t(sum_loss_cl)}")
            tf.print(f"\nTrain x-y loss:             {_to_output_t(sum_loss_xy)}")
            tf.print(f"\nTrain w-h loss:             {_to_output_t(sum_loss_wh)}")
            tf.print(f"\n")
            tf.print(f"\nValidation total loss:           {_to_output_v(val_loss)}")
            tf.print(f"\nValidation (no-)objectness loss: {_to_output_v(val_loss_noobj)}")
            tf.print(f"\nValidation objectness loss:      {_to_output_v(val_loss_obj)}")
            tf.print(f"\nValidation classification loss:  {_to_output_v(val_loss_cl)}")
            tf.print(f"\nValidation x-y loss:             {_to_output_v(val_loss_xy)}")
            tf.print(f"\nValidation w-h loss:             {_to_output_v(val_loss_wh)}")
            tf.print(f"\n===================================================================================================================\n")

        for epoch in range(self.next_train_epoch, epochs, 1):

            try:

                if progbar:
                    progbar_output = tf.keras.utils.Progbar(TRAIN_BATCH_CNT)
                tf.print(f"\nEpoch {epoch}:")

                # loss stats variables

                sum_loss = 0
                sum_loss_noobj = 0
                sum_loss_obj = 0
                sum_loss_cl = 0
                sum_loss_xy = 0
                sum_loss_wh = 0

                val_loss = 0
                val_loss_noobj = 0
                val_loss_obj = 0
                val_loss_cl = 0
                val_loss_xy = 0
                val_loss_wh = 0

                # train loop

                batch_idx = 0
                for (imgs, obj_mask_size1, ignored_mask_size1, target_mask_size1, \
                            obj_mask_size2, ignored_mask_size2, target_mask_size2, \
                            obj_mask_size3, ignored_mask_size3, target_mask_size3, \
                            gt_boxes) \
                    in self.data_loader.load_data(TRAIN_BATCH_SIZE, "train", shuffle=True, augment_probability=0.7):

                    new_lr = self.lr_scheduler(epoch, batch_idx, self.full_network.optimizer.learning_rate)
                    self.full_network.optimizer.learning_rate = new_lr

                    with tf.GradientTape() as tape:

                        out_s1, out_s2, out_s3 = self.full_network(imgs, training=True)

                        loss_value, noobj, obj, cl, xy, wh = yolov3_loss_perscale(out_s1, obj_mask_size1, ignored_mask_size1, target_mask_size1, anchors_relative[0], gt_boxes)
                        loss_value_, noobj_, obj_, cl_, xy_, wh_ = yolov3_loss_perscale(out_s2, obj_mask_size2, ignored_mask_size2, target_mask_size2, anchors_relative[1], gt_boxes)
                        loss_value__, noobj__, obj__, cl__, xy__, wh__ = yolov3_loss_perscale(out_s3, obj_mask_size3, ignored_mask_size3, target_mask_size3, anchors_relative[2], gt_boxes)

                        loss_value += loss_value_ + loss_value__
                        noobj += noobj_ + noobj__
                        obj += obj_ + obj__
                        cl += cl_ + cl__
                        xy += xy_ + xy__
                        wh += wh_ + wh__

                    if tf.reduce_sum(tf.cast(tf.math.is_nan(loss_value), tf.int32)) > 0:
                        tf.print(f"loss became nan at epoch {epoch} batch idx {batch_idx}")
                        return "nan found"

                    gradients = tape.gradient(loss_value, self.full_network.trainable_weights)
                    self.full_network.optimizer.apply_gradients(zip(gradients, self.full_network.trainable_weights))

                    batch_idx += 1
                    if progbar:
                        progbar_output.update(batch_idx)

                    sum_loss += loss_value
                    sum_loss_noobj += noobj
                    sum_loss_obj += obj
                    sum_loss_cl += cl
                    sum_loss_xy += xy
                    sum_loss_wh += wh

                    #break

                #continue

                # validation loop

                for (imgs, obj_mask_size1, ignored_mask_size1, target_mask_size1, \
                            obj_mask_size2, ignored_mask_size2, target_mask_size2, \
                            obj_mask_size3, ignored_mask_size3, target_mask_size3, \
                            gt_boxes) \
                    in self.data_loader.load_data(VALIDATION_BATCH_SIZE, "validation", augment_probability=0):
                    
                    out_s1, out_s2, out_s3 = self.full_network(imgs, training=False)

                    loss_value_, noobj_, obj_, cl_, xy_, wh_ = yolov3_loss_perscale(out_s1, obj_mask_size1, ignored_mask_size1, target_mask_size1, anchors_relative[0], gt_boxes)
                    val_loss += loss_value_
                    val_loss_noobj += noobj_
                    val_loss_obj += obj_
                    val_loss_cl += cl_
                    val_loss_xy += xy_
                    val_loss_wh += wh_

                    loss_value_, noobj_, obj_, cl_, xy_, wh_ = yolov3_loss_perscale(out_s2, obj_mask_size2, ignored_mask_size2, target_mask_size2, anchors_relative[1], gt_boxes)
                    val_loss += loss_value_
                    val_loss_noobj += noobj_
                    val_loss_obj += obj_
                    val_loss_cl += cl_
                    val_loss_xy += xy_
                    val_loss_wh += wh_
                    
                    loss_value_, noobj_, obj_, cl_, xy_, wh_ = yolov3_loss_perscale(out_s3, obj_mask_size3, ignored_mask_size3, target_mask_size3, anchors_relative[2], gt_boxes)
                    val_loss += loss_value_
                    val_loss_noobj += noobj_
                    val_loss_obj += obj_
                    val_loss_cl += cl_
                    val_loss_xy += xy_
                    val_loss_wh += wh_

                if tf.reduce_sum(tf.cast(tf.math.is_nan(val_loss), tf.int32)) > 0:
                    tf.print(f"val loss became nan at epoch {epoch}")
                    return "nan found"

            except KeyboardInterrupt:

                if save_on_keyboard_interrupt:

                    tf.print(f"\nTraining paused at epoch {epoch}.")

                    self.next_train_epoch = epoch
                    self.cache_manager.store_model()

                    stats = [train_loss_stats, train_loss_stats_noobj, train_loss_stats_obj,
                            train_loss_stats_cl, train_loss_stats_xy, train_loss_stats_wh,
                            validation_loss_stats, validation_loss_stats_noobj, validation_loss_stats_obj,
                            validation_loss_stats_cl, validation_loss_stats_xy, validation_loss_stats_wh]

                    self.cache_manager.store_train_stats(stats)

                else:
                    tf.print(f"\nTraining stopped at epoch {epoch} - no checkpoint occured.")

                return

            _log_show_losses()

            if checkpoint_sched(epoch, sum_loss, val_loss):
                
                self.next_train_epoch = epoch + 1
                self.cache_manager.store_model()

                stats = [train_loss_stats, train_loss_stats_noobj, train_loss_stats_obj,
                            train_loss_stats_cl, train_loss_stats_xy, train_loss_stats_wh,
                            validation_loss_stats, validation_loss_stats_noobj, validation_loss_stats_obj,
                            validation_loss_stats_cl, validation_loss_stats_xy, validation_loss_stats_wh]

                self.cache_manager.store_train_stats(stats)

                if copy_at_checkpoint:
                    vloss = _to_output_v(val_loss)
                    self.cache_manager.copy_model(f"{self.cache_manager.cache_idx}_e{epoch}_vloss{vloss}")

        if self.next_train_epoch >= epochs:
            tf.print(f"\nModel is already trained (at least) {epochs} epochs.")

        else:
            tf.print(f"\nTraining is done ({epochs} epochs).")

            self.next_train_epoch = epochs
            self.cache_manager.store_model()

            stats = [train_loss_stats, train_loss_stats_noobj, train_loss_stats_obj,
                    train_loss_stats_cl, train_loss_stats_xy, train_loss_stats_wh,
                    validation_loss_stats, validation_loss_stats_noobj, validation_loss_stats_obj,
                    validation_loss_stats_cl, validation_loss_stats_xy, validation_loss_stats_wh]

            self.cache_manager.store_train_stats(stats)

    def show_architecture_stats(self):
        
        self.full_network.summary()
        tf.keras.utils.plot_model(self.full_network, show_shapes=True)

    def stage(self):

        if self._status is Network.NOT_CREATED:
            tf.print("Network has not yet been initialized.")

        elif self._status is Network.UNTRAINED:
            tf.print("Network is not trained.")

        elif self._status is Network.TRAINING_ENCODER:
            tf.print(f"The encoder has been trained (for classification) {self.next_train_epoch} epochs.")

        elif self._status is Network.TRAINING_DETECTION:
            tf.print(f"Full network has been trained (for detection) {self.next_train_epoch} epochs.")

    def predict(self, subset="validation", obj_threshold=0.6, nms_threshold=0.6):
        
        if self._status < Network.TRAINING_DETECTION:
            tf.print("Network not yet initialized")
            return

        anchors_relative = [tf.cast(GRID_CELL_CNT[d] * (self.data_loader.anchors[d] / IMG_SIZE[0]), dtype=tf.float32) for d in range(SCALE_CNT)]

        for (img, _, _, _, _, _, _, _, _, _, _) in self.data_loader.load_data(1, subset, shuffle=True, augment_probability=0):

            output = self.full_network(img, training=False)
            pred_xy_min, pred_xy_max, pred_class, pred_class_p = self.stats_manager.parse_prediction(output, anchors_relative, obj_threshold, nms_threshold)

            self.stats_manager.show_prediction(np.array(((img[0] + 1.0) / 2.0) * 255.0, dtype=np.uint8), \
                                                pred_xy_min, pred_xy_max, pred_class, pred_class_p)

    def compute_precision_recall_stats(self, subset="validation", nms_threshold=0.6, overwrite_old_stats=False):

        if self._status < Network.TRAINING_DETECTION:
            tf.print("Network not yet initialized")
            return

        if self.stats_computed and (overwrite_old_stats is False):
            tf.print("Validation stats (tp, fp, fn) are already computed.")
            return

        def _get_keys():
            ks = list(self.data_loader.imgs[subset].keys())
            yield from ks

        anchors_relative = [tf.cast(GRID_CELL_CNT[d] * (self.data_loader.anchors[d] / IMG_SIZE[0]), dtype=tf.float32) for d in range(SCALE_CNT)]

        ks = _get_keys()

        tf.print(f"Computing stats...")

        cnt_ = 0
        TOTAL_IMG_CNT = self.data_loader.get_img_cnt("validation")
        progbar = tf.keras.utils.Progbar(TOTAL_IMG_CNT)

        for (img, _, _, _, _, _, _, _, _, _, _) in self.data_loader.load_data(1, subset, shuffle=False, augment_probability=0):

            k = next(ks)
            output = self.full_network(img, training=False)

            self.stats_manager.update_tp_fp_fn(output, self.data_loader.imgs[subset][k]["objs"], anchors_relative, nms_threshold)

            progbar.update(cnt_)
            cnt_ += 1

        self.stats_computed = True
        self.cache_manager.store_stats_manager()

    def get_ap(self, threshold):
        return self.stats_manager.get_ap(threshold)

    def get_mean_ap(self):
        return self.stats_manager.get_mean_ap()

    def get_precision_recall(self, obj_thresh=0.6, iou_thresh=0.6):

        tp = 0
        tp_fp = 0
        tp_fn = 0
        for pr_dict_perclass in self.stats_manager.pr_dict.values():
            tp += pr_dict_perclass[iou_thresh][obj_thresh]["tp"]
            tp_fp += pr_dict_perclass[iou_thresh][obj_thresh]["tp_fp"]
            tp_fn += pr_dict_perclass[iou_thresh][obj_thresh]["tp_fn"]

        return tp / tp_fp, tp / tp_fn

class NetworkCacheManager:

    def __init__(self, network: Network, cache_key, cache_idx):
        
        self.network = network
        '''
            the network to which this cache manager is assigned to
        '''

        if cache_idx is not None:

            assert(cache_key is not None)

            self.cache_key = cache_key
            '''
                cache key used for saving/loading a model
            '''

            self.cache_idx = cache_idx
            '''
                to be able to use the same data cache for multiple model savings, a cache "subkey" is also used
            '''

        else:

            self.cache_key = None
            self.cache_idx = None

    def copy_model(self, new_cache_idx):

        assert(new_cache_idx != TMP_CACHE_KEY)

        if self.cache_key is None:
            tf.print("Cannot copy when current model is not cached")

        assert(new_cache_idx != self.cache_idx)

        with open(f"{MODEL_CACHE_PATH}{self.cache_key}_{self.cache_idx}_opt", "rb") as opt_f:
            with open(f"{MODEL_CACHE_PATH}{self.cache_key}_{new_cache_idx}_opt", "wb+") as opt_f_:

                opt_w = opt_f.read()
                opt_f_.write(opt_w)

        with open(f"{MODEL_CACHE_PATH}{self.cache_key}_{self.cache_idx}_pretrain_opt", "rb") as opt_f:
            with open(f"{MODEL_CACHE_PATH}{self.cache_key}_{new_cache_idx}_pretrain_opt", "wb+") as opt_f_:

                opt_w = opt_f.read()
                opt_f_.write(opt_w)

        with open(f"{MODEL_CACHE_PATH}{self.cache_key}_{self.cache_idx}_status.json", "r") as last_epoch_f:
            with open(f"{MODEL_CACHE_PATH}{self.cache_key}_{new_cache_idx}_status.json", "w+") as last_epoch_f_:

                training_epoch = last_epoch_f.read()
                last_epoch_f_.write(training_epoch)

        full_network = tf.keras.models.load_model(f"{MODEL_CACHE_PATH}{self.cache_key}_{self.cache_idx}_model", custom_objects={
                                                                                                                                "ConvLayer": ConvLayer,
                                                                                                                                "ResBlock": ResBlock, 
                                                                                                                                "ResSequence": ResSequence
                                                                                                                                }
                                                        )
        tf.keras.models.save_model(full_network, f"{MODEL_CACHE_PATH}{self.cache_key}_{new_cache_idx}_model", overwrite=True)

        encoder = tf.keras.models.load_model(f"{MODEL_CACHE_PATH}{self.cache_key}_{self.cache_idx}_pretrain_model", custom_objects={
                                                                                                                                    "ConvLayer": ConvLayer,
                                                                                                                                    "ResBlock": ResBlock, 
                                                                                                                                    "ResSequence": ResSequence
                                                                                                                                    }
                                                        )
        tf.keras.models.save_model(encoder, f"{MODEL_CACHE_PATH}{self.cache_key}_{new_cache_idx}_pretrain_model", overwrite=True)

        try:

            with open(f"{TRAIN_STATS_PATH}{self.cache_key}_{self.cache_idx}_stats", "rb") as stats_f:
                with open(f"{TRAIN_STATS_PATH}{self.cache_key}_{new_cache_idx}_stats", "wb+") as stats_f_:

                    stats = stats_f.read()
                    stats_f_.write(stats)

        except FileNotFoundError:
            pass

        try:

            with open(f"{TRAIN_STATS_PATH}{self.cache_key}_{self.cache_idx}_pretrain_stats", "rb") as stats_f:
                with open(f"{TRAIN_STATS_PATH}{self.cache_key}_{new_cache_idx}_pretrain_stats", "wb+") as stats_f_:

                    stats = stats_f.read()
                    stats_f_.write(stats)

        except FileNotFoundError:
            pass

        try:

            with open(f"{VALIDATION_STATS_PATH}{self.cache_key}_{self.cache_idx}_stats_manager", "rb") as stats_f:
                with open(f"{VALIDATION_STATS_PATH}{self.cache_key}_{new_cache_idx}_stats_manager", "wb+") as stats_f_:

                    stats = stats_f.read()
                    stats_f_.write(stats)

        except FileNotFoundError:
            pass

        tf.print(f"Model copied from idx {self.cache_idx} to idx {new_cache_idx}.")

    def _init_optimizer(self, net):
        '''
            hack to initialize weights for the optimizer of a network, 
            in case they are not already initialized
            https://github.com/keras-team/keras/issues/15298
        '''

        ws = net.trainable_weights
        noop = [tf.zeros_like(w) for w in ws]
        net.optimizer.apply_gradients(zip(noop, ws))

    def get_model(self):
        '''
            method for loading:
            * the saved model
            * its optimizer
            * the last executed epoch
        '''

        if self.cache_key is not None:

            try:

                with open(f"{MODEL_CACHE_PATH}{self.cache_key}_{self.cache_idx}_status.json", "r") as status_f:
                    status = json.load(status_f)

                self.network.next_train_epoch = status["next_epoch"]
                self.network.next_pretrain_epoch = status["next_pretrain_epoch"]
                self.network._status = status["state"]

                self.network.full_network = tf.keras.models.load_model(f"{MODEL_CACHE_PATH}{self.cache_key}_{self.cache_idx}_model", custom_objects={
                                                                                                                                                    "ConvLayer": ConvLayer,
                                                                                                                                                    "ResBlock": ResBlock, 
                                                                                                                                                    "ResSequence": ResSequence
                                                                                                                                                    }
                                                                )
                                                                
                self.network.encoder = tf.keras.models.load_model(f"{MODEL_CACHE_PATH}{self.cache_key}_{self.cache_idx}_pretrain_model", custom_objects={
                                                                                                                                                        "ConvLayer": ConvLayer,
                                                                                                                                                        "ResBlock": ResBlock, 
                                                                                                                                                        "ResSequence": ResSequence
                                                                                                                                                        }
                                                                    )

                with open(f"{MODEL_CACHE_PATH}{self.cache_key}_{self.cache_idx}_opt", "rb") as opt_f:
                    opt_w = opt_f.read()
                    opt_w = pickle.loads(opt_w)

                self._init_optimizer(self.network.full_network)
                self.network.full_network.optimizer.set_weights(opt_w)

                with open(f"{MODEL_CACHE_PATH}{self.cache_key}_{self.cache_idx}_pretrain_opt", "rb") as opt_f:
                    opt_w = opt_f.read()
                    opt_w = pickle.loads(opt_w)

                self._init_optimizer(self.network.encoder)
                self.network.encoder.optimizer.set_weights(opt_w)

                tf.print(f"Model with cache key {self.cache_key} (idx {self.cache_idx}) has been found and loaded.")
                
            except FileNotFoundError:
                pass

    def store_model(self):
        '''
            method for saving a model, along with its optimizer
            the model is saved automatically (if cache is used):
            * in build_components(), if the model is new
            * in train, at the end
            * in train, if it is stopped with Ctrl-C
        '''

        if self.network._status < Network.TRAINING_ENCODER:
            tf.print("Network is not trained. Nothing to save.")
            return

        if self.cache_idx is not None:

            status =    {
                        "next_epoch": self.network.next_train_epoch,
                        "next_pretrain_epoch": self.network.next_pretrain_epoch,
                        "state": self.network._status
                        }

            with open(f"{MODEL_CACHE_PATH}{self.cache_key}_{self.cache_idx}_status.json", "w+") as status_f:
                json.dump(status, status_f)
    
            self._init_optimizer(self.network.full_network)
            opt_w = tf.keras.backend.batch_get_value(self.network.full_network.optimizer.weights)
            with open(f"{MODEL_CACHE_PATH}{self.cache_key}_{self.cache_idx}_opt", "wb+") as opt_f:
                opt_w = pickle.dumps(opt_w)
                opt_f.write(opt_w)

            self._init_optimizer(self.network.encoder)
            opt_w = tf.keras.backend.batch_get_value(self.network.encoder.optimizer.weights)
            with open(f"{MODEL_CACHE_PATH}{self.cache_key}_{self.cache_idx}_pretrain_opt", "wb+") as opt_f:
                opt_w = pickle.dumps(opt_w)
                opt_f.write(opt_w)

            tf.keras.models.save_model(self.network.full_network, f"{MODEL_CACHE_PATH}{self.cache_key}_{self.cache_idx}_model", overwrite=True)
            tf.keras.models.save_model(self.network.encoder, f"{MODEL_CACHE_PATH}{self.cache_key}_{self.cache_idx}_pretrain_model", overwrite=True)

            tf.print(f"Model with key {self.cache_key} (idx {self.cache_idx}) has been saved.")

    def get_train_stats(self):
        '''
            loads the train statistics under the cache_key, cache_idx entry
        '''

        if self.cache_key is not None:
            cache_key = self.cache_key
            cache_idx = self.cache_idx

        else:
            cache_key = TMP_CACHE_KEY
            cache_idx = TMP_CACHE_KEY

        try:
            
            with open(f"{TRAIN_STATS_PATH}{cache_key}_{cache_idx}_stats", "rb") as stats_f:
                
                stats = stats_f.read()
                stats = pickle.loads(stats)

            return stats

        except FileNotFoundError:
            return [[] for _ in range(12)]

    def get_pretrain_stat(self):
        '''
            loads the train statistics under the cache_key, cache_idx entry
        '''

        if self.cache_key is not None:
            cache_key = self.cache_key
            cache_idx = self.cache_idx

        else:
            cache_key = TMP_CACHE_KEY
            cache_idx = TMP_CACHE_KEY

        try:
            
            with open(f"{TRAIN_STATS_PATH}{cache_key}_{cache_idx}_pretrain_stats", "rb") as stats_f:
                
                stats = stats_f.read()
                stats = pickle.loads(stats)

            return stats

        except FileNotFoundError:
            return [[] for _ in range(4)]

    def store_train_stats(self, stats):
        '''
            saves the train statistics under the cache_key, cache_idx entry
        '''

        if self.cache_key is not None:
            cache_key = self.cache_key
            cache_idx = self.cache_idx

        else:
            cache_key = TMP_CACHE_KEY
            cache_idx = TMP_CACHE_KEY

        with open(f"{TRAIN_STATS_PATH}{cache_key}_{cache_idx}_stats", "wb+") as stats_f:
            
            stats = pickle.dumps(stats)
            stats_f.write(stats)

    def store_pretrain_stats(self, stats):
        '''
            saves the pre-train statistics under the cache_key, cache_idx entry
        '''

        if self.cache_key is not None:
            cache_key = self.cache_key
            cache_idx = self.cache_idx

        else:
            cache_key = TMP_CACHE_KEY
            cache_idx = TMP_CACHE_KEY

        with open(f"{TRAIN_STATS_PATH}{cache_key}_{cache_idx}_pretrain_stats", "wb+") as stats_f:
            
            stats = pickle.dumps(stats)
            stats_f.write(stats)

    def store_train_stats_fig(self):

        if self.cache_key is not None:
            cache_key = self.cache_key
            cache_idx = self.cache_idx

        else:
            cache_key = TMP_CACHE_KEY
            cache_idx = TMP_CACHE_KEY

        plt.savefig(f"{TRAIN_STATS_PATH}{cache_key}_{cache_idx}_stats_plot")

    def store_pretrain_stats_fig(self):

        if self.cache_key is not None:
            cache_key = self.cache_key
            cache_idx = self.cache_idx

        else:
            cache_key = TMP_CACHE_KEY
            cache_idx = TMP_CACHE_KEY

        plt.savefig(f"{TRAIN_STATS_PATH}{cache_key}_{cache_idx}_pretrain_stats_plot")

    def get_stats_manager(self):

        try:

            with open(f"{VALIDATION_STATS_PATH}{self.cache_key}_{self.cache_idx}_stats_manager", "rb") as stats_f:
                stats_manager = pickle.loads(stats_f.read())
            
            return stats_manager, True

        except FileNotFoundError:
            return StatsManager(self.network.data_loader.onehot_to_name, \
                                iou_thresholds=[thr / 100 for thr in range(0, 100, 5)], \
                                confidence_thresholds=[thr / 100 for thr in range(10, 100, 5)]), False

    def store_stats_manager(self):

        if self.cache_key is not None:
        
            with open(f"{VALIDATION_STATS_PATH}{self.cache_key}_{self.cache_idx}_stats_manager", "wb+") as stats_f:
                stats_manager = pickle.dumps(self.network.stats_manager)
                stats_f.write(stats_manager)
                