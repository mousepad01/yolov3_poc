from math import floor
import numpy as np
import tensorflow as tf
import pickle

from custom_keras import *
from utils import *
from loss import *

from predictions import *
from data_processing import *

class Network:

    def __init__(self, data_loader, cache_idx=None):

        self.data_loader: DataLoader = data_loader
        '''
            contains data info and all that stuff
        '''
        
        if cache_idx is not None:

            assert(self.data_loader.cache_key is not None)

            self.cache_key = self.data_loader.cache_key
            '''
                cache key used for saving/loading a model
            '''

            self.cache_idx = cache_idx
            '''
                to be able to use the same data cache for multiple model savings, a cache "subkey" is also used
            '''

            self.next_training_epoch = 0
            '''
                the last training epoch that was executing when the model had been saved
            '''

        else:

            self.cache_key = None
            self.cache_idx = None
            self.next_training_epoch = 0

        self.backbone: tf.keras.Model = None
        '''
            backbone for feature extraction
        '''

        self.input_layer: tf.keras.layer = None
        '''
            416 x 416 x 3
        '''

        self.full_network: tf.keras.Model = None
        '''
            includes the full network for object detection (so, everything except backbone classification head)
        '''

        self.lr_scheduler: function = None
        '''
            schedule the learning rate during training
            * NOTE: this function is not saved when the training is interrupred; it is best this function is defined stateless
        '''

    def copy_model(self, new_cache_idx):
        '''
            Copy model cache and its stats under new cache_idx (same cache_key)
        '''

        assert(new_cache_idx != TMP_CACHE_KEY)

        if self.cache_key is None:
            tf.print("Cannot copy when current model is not cached")

        assert(new_cache_idx != self.cache_idx)

        with open(f"{MODEL_CACHE_PATH}{self.cache_key}_{self.cache_idx}_opt", "rb") as opt_f:
            with open(f"{MODEL_CACHE_PATH}{self.cache_key}_{new_cache_idx}_opt", "wb+") as opt_f_:

                opt_w = opt_f.read()
                opt_f_.write(opt_w)

        with open(f"{MODEL_CACHE_PATH}{self.cache_key}_{self.cache_idx}_next_epoch", "r") as last_epoch_f:
            with open(f"{MODEL_CACHE_PATH}{self.cache_key}_{new_cache_idx}_next_epoch", "w+") as last_epoch_f_:

                training_epoch = last_epoch_f.read()
                last_epoch_f_.write(training_epoch)

        full_network = tf.keras.models.load_model(f"{MODEL_CACHE_PATH}{self.cache_key}_{self.cache_idx}_model", custom_objects={
                                                                                                                                "ConvLayer": ConvLayer,
                                                                                                                                "ResBlock": ResBlock, 
                                                                                                                                "ResSequence": ResSequence
                                                                                                                                }
                                                        )
        tf.keras.models.save_model(full_network, f"{MODEL_CACHE_PATH}{self.cache_key}_{new_cache_idx}_model", overwrite=True)

        with open(f"{TRAIN_STATS_PATH}{self.cache_key}_{self.cache_idx}_stats", "rb") as stats_f:
            with open(f"{TRAIN_STATS_PATH}{self.cache_key}_{new_cache_idx}_stats", "wb+") as stats_f_:

                stats = stats_f.read()
                stats_f_.write(stats)

        tf.print(f"Model copied from idx {self.cache_idx} to idx {new_cache_idx}.")

    # move to cache loader
    def _load_model(self):
        '''
            internal method for loading:
            * the saved model
            * its optimizer
            * the last executed epoch
        '''

        with open(f"{MODEL_CACHE_PATH}{self.cache_key}_{self.cache_idx}_opt", "rb") as opt_f:
            opt_w = opt_f.read()
            opt_w = pickle.loads(opt_w)

        self.full_network = tf.keras.models.load_model(f"{MODEL_CACHE_PATH}{self.cache_key}_{self.cache_idx}_model", custom_objects={
                                                                                                                                        "ConvLayer": ConvLayer,
                                                                                                                                        "ResBlock": ResBlock, 
                                                                                                                                        "ResSequence": ResSequence
                                                                                                                                        }
                                                        )

        # hack to initialize weights for the optimizer, so that old ones can be loaded 
        # https://github.com/keras-team/keras/issues/15298
        ws = self.full_network.trainable_weights
        noop = [tf.zeros_like(w) for w in ws]
        self.full_network.optimizer.apply_gradients(zip(noop, ws))

        self.full_network.optimizer.set_weights(opt_w)

        with open(f"{MODEL_CACHE_PATH}{self.cache_key}_{self.cache_idx}_next_epoch", "r") as last_epoch_f:
            self.next_training_epoch = int(last_epoch_f.read())

        tf.print(f"Model with cache key {self.cache_key} (idx {self.cache_idx}) has been found and loaded, along with its optimizer and the last training epoch.")

    # move to cache saver
    def _save_model(self, last_epoch):
        '''
            internal method for saving a model, along with its optimizer
            the model is saved automatically (if cache is used):
            * in build_components(), if the model is new
            * in train, at the end
            * in train, if it is stopped with Ctrl-C
        '''
        
        self.next_training_epoch = last_epoch
        with open(f"{MODEL_CACHE_PATH}{self.cache_key}_{self.cache_idx}_next_epoch", "w+") as last_epoch_f:
            last_epoch_f.write(f"{self.next_training_epoch}")
 
        opt_w = tf.keras.backend.batch_get_value(self.full_network.optimizer.weights)
        with open(f"{MODEL_CACHE_PATH}{self.cache_key}_{self.cache_idx}_opt", "wb+") as opt_f:
            opt_w = pickle.dumps(opt_w)
            opt_f.write(opt_w)

        tf.keras.models.save_model(self.full_network, f"{MODEL_CACHE_PATH}{self.cache_key}_{self.cache_idx}_model", overwrite=True)

        tf.print(f"Model with key {self.cache_key} (idx {self.cache_idx}) has been saved.")

    # can use cache loader
    def build_components(self, optimizer: tf.keras.optimizers.Optimizer, lr_scheduler=lambda epoch, lr: lr, backbone="darknet-53"):
        ''' 
            if there is already a saved model and cache is used, it loads that model
            if not, it builds the model  using the given parameters
            * optimizer: to use during training (ignored if a saved model is loaded)
            * lr_scheduler(current epoch, current lr) => updated lr
            * backbone: string with the name of the backbone to use (ignored if a saved model is loaded)
        '''

        self.lr_scheduler = lr_scheduler

        if self.cache_key is not None:

            try:
                
                # check whether the cache already exists
                with open(f"{MODEL_CACHE_PATH}{self.cache_key}_{self.cache_idx}_next_epoch", "r") as last_epoch_f:
                    pass

                self._load_model()
                return
                
            except FileNotFoundError:
                pass

        tf.print("Building a new model...")

        CLASS_COUNT = len(self.data_loader.used_categories)

        if backbone == "darknet-53":
        
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
            output_scale1 = ConvLayer(ANCHOR_PERSCALE_CNT * (4 + 1 + CLASS_COUNT), 1)(conv_scale1_6)

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
            output_scale2 = ConvLayer(ANCHOR_PERSCALE_CNT * (4 + 1 + CLASS_COUNT), 1)(conv_scale2_6)

            # output for scale 3

            conv_scale23 = ConvLayer(128, 1)(conv_scale2_5)
            upsample_scale23 = tf.keras.layers.UpSampling2D((2, 2))(conv_scale23)
            features_scale3 = tf.keras.layers.Concatenate(axis=-1)([res_back3, upsample_scale23])

            conv_scale3_1 = ConvLayer(128, 1)(features_scale3)
            conv_scale3_2 = ConvLayer(256, 3)(conv_scale3_1)
            conv_scale3_3 = ConvLayer(128, 1)(conv_scale3_2)
            conv_scale3_4 = ConvLayer(256, 3)(conv_scale3_3)
            conv_scale3_5 = ConvLayer(128, 1)(conv_scale3_4)

            conv_scale3_6 = ConvLayer(256, 3)(conv_scale3_5)
            output_scale3 = ConvLayer(ANCHOR_PERSCALE_CNT * (4 + 1 + CLASS_COUNT), 1)(conv_scale3_6)

            self.full_network = tf.keras.Model(inputs=input_img, outputs=[output_scale1, output_scale2, output_scale3])

        elif backbone == "small":

            # the backbone
            input_img = tf.keras.layers.Input((IMG_SIZE[0], IMG_SIZE[1], 3))

            conv_back1 = ConvLayer(16, 3)(input_img) # 416
            res_back1 = ResSequence(32, 1)(conv_back1) # 208
            res_back2 = ResSequence(64, 2)(res_back1) # 104
            res_back3 = ResSequence(128, 4)(res_back2) # 52
            res_back4 = ResSequence(256, 4)(res_back3) # 26
            res_back5 = ResSequence(512, 2)(res_back4) # 13

            self.backbone = tf.keras.Model(inputs=input_img, outputs=res_back5)

            # the entire network

            # output for scale 1

            features_scale1 = res_back5

            conv_scale1_1 = ConvLayer(256, 1)(features_scale1)
            conv_scale1_2 = ConvLayer(512, 3)(conv_scale1_1)
            #conv_scale1_3 = ConvLayer(256, 1)(conv_scale1_2)
            #conv_scale1_4 = ConvLayer(512, 3)(conv_scale1_3)
            conv_scale1_5 = ConvLayer(256, 1)(conv_scale1_2)

            conv_scale1_6 = ConvLayer(512, 3)(conv_scale1_5)
            output_scale1 = ConvLayer(ANCHOR_PERSCALE_CNT * (4 + 1 + CLASS_COUNT), 1)(conv_scale1_6)

            # output for scale 2

            conv_scale12 = ConvLayer(128, 1)(conv_scale1_5)
            upsample_scale12 = tf.keras.layers.UpSampling2D((2, 2))(conv_scale12)
            features_scale2 = tf.keras.layers.Concatenate(axis=-1)([res_back4, upsample_scale12])

            conv_scale2_1 = ConvLayer(128, 1)(features_scale2)
            conv_scale2_2 = ConvLayer(256, 3)(conv_scale2_1)
            #conv_scale2_3 = ConvLayer(128, 1)(conv_scale2_2)
            #conv_scale2_4 = ConvLayer(256, 3)(conv_scale2_3)
            conv_scale2_5 = ConvLayer(128, 1)(conv_scale2_2)

            conv_scale2_6 = ConvLayer(256, 3)(conv_scale2_5)
            output_scale2 = ConvLayer(ANCHOR_PERSCALE_CNT * (4 + 1 + CLASS_COUNT), 1)(conv_scale2_6)

            # output for scale 3

            conv_scale23 = ConvLayer(64, 1)(conv_scale2_5)
            upsample_scale23 = tf.keras.layers.UpSampling2D((2, 2))(conv_scale23)
            features_scale3 = tf.keras.layers.Concatenate(axis=-1)([res_back3, upsample_scale23])

            conv_scale3_1 = ConvLayer(64, 1)(features_scale3)
            conv_scale3_2 = ConvLayer(128, 3)(conv_scale3_1)
            #conv_scale3_3 = ConvLayer(64, 1)(conv_scale3_2)
            #conv_scale3_4 = ConvLayer(128, 3)(conv_scale3_3)
            conv_scale3_5 = ConvLayer(64, 1)(conv_scale3_2)

            conv_scale3_6 = ConvLayer(128, 3)(conv_scale3_5)
            output_scale3 = ConvLayer(ANCHOR_PERSCALE_CNT * (4 + 1 + CLASS_COUNT), 1)(conv_scale3_6)

            self.full_network = tf.keras.Model(inputs=input_img, outputs=[output_scale1, output_scale2, output_scale3])

        else:
            print("unknown backbone")
            quit()
        
        self.full_network.compile(optimizer=optimizer)

        if self.cache_idx is not None:
            self._save_model(0)

    # can use cache loader (self passed as arg)
    # can use cache saver
    def plot_train_stats(self, show_on_screen=False, save_image=True):
        '''
            loads AND shows the train statistics under the cache_key, cache_idx entry
            * show_on_screen: if True, show on screen
            * save_image: if True, save as image in the TRAIN_STATS_PATH
        '''

        if self.cache_key is not None:
            cache_key = self.cache_key
            cache_idx = self.cache_idx

        else:
            cache_key = TMP_CACHE_KEY
            cache_idx = TMP_CACHE_KEY
            
        with open(f"{TRAIN_STATS_PATH}{cache_key}_{cache_idx}_stats", "rb") as stats_f:
            
            stats = stats_f.read()
            stats = pickle.loads(stats)

        train_loss_stats, train_loss_stats_noobj, train_loss_stats_obj, \
        train_loss_stats_cl, train_loss_stats_xy, train_loss_stats_wh, \
        validation_loss_stats, validation_loss_stats_noobj, validation_loss_stats_obj, \
        validation_loss_stats_cl, validation_loss_stats_xy, validation_loss_stats_wh = stats

        _, ax = plt.subplots(3, 2)

        ax[0][0].plot([idx for idx in range(len(train_loss_stats))],
                        train_loss_stats)
        ax[0][0].plot([idx for idx in range(len(validation_loss_stats))],
                        validation_loss_stats)
        ax[0][0].grid(True)
        ax[0][0].set_title("total loss")

        ax[0][1].plot([idx for idx in range(len(train_loss_stats))],
                        train_loss_stats_noobj)
        ax[0][1].plot([idx for idx in range(len(validation_loss_stats))],
                        validation_loss_stats_noobj)
        ax[0][1].grid(True)
        ax[0][1].set_title("(no-)objectness loss")

        ax[1][0].plot([idx for idx in range(len(train_loss_stats))],
                        train_loss_stats_obj)
        ax[1][0].plot([idx for idx in range(len(validation_loss_stats))],
                        validation_loss_stats_obj)
        ax[1][0].grid(True)
        ax[1][0].set_title("objectness loss")

        ax[1][1].plot([idx for idx in range(len(train_loss_stats))],
                        train_loss_stats_cl)
        ax[1][1].plot([idx for idx in range(len(validation_loss_stats))],
                        validation_loss_stats_cl)
        ax[1][1].grid(True)
        ax[1][1].set_title("classification loss")

        ax[2][0].plot([idx for idx in range(len(train_loss_stats))],
                        train_loss_stats_xy)
        ax[2][0].plot([idx for idx in range(len(validation_loss_stats))],
                        validation_loss_stats_xy)
        ax[2][0].grid(True)
        ax[2][0].set_title("x-y loss")

        ax[2][1].plot([idx for idx in range(len(train_loss_stats))],
                        train_loss_stats_wh)
        ax[2][1].plot([idx for idx in range(len(validation_loss_stats))],
                        validation_loss_stats_wh)
        ax[2][1].grid(True)
        ax[2][1].set_title("w-h loss")

        if save_image:
            plt.savefig(f"{TRAIN_STATS_PATH}{cache_key}_{cache_idx}_stats_plot")

        if show_on_screen:
            plt.show()

    # can use cache loader
    def _load_train_stats(self):
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

    # move to cache saver
    def _save_train_stats(self, stats):
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
            
    # can use cache loader / saver
    def train(self, epochs, batch_size, checkpoint_sched=lambda epoch, loss, vloss: False):
        '''
            * epochs: number of total epochs (effective number of epochs executed: epochs - self.next_training_epoch + 1)
            * batch_size: for training
            * checkpoint_sched: decide whether to create a checkpoint, after the end of an epoch
        '''

        if self.full_network is None:
            tf.print("Network not yet initialized")
            return

        TRAIN_BATCH_SIZE = batch_size
        VALIDATION_BATCH_SIZE = DATA_LOAD_BATCH_SIZE

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
        validation_loss_stats_cl, validation_loss_stats_xy, validation_loss_stats_wh = self._load_train_stats()

        def _log_show_losses():

            def _to_output_t(x): 
                return floor((x / TRAIN_IMG_CNT) * (10 ** LOSS_OUTPUT_PRECISION)) / (10 ** LOSS_OUTPUT_PRECISION)

            def _to_output_v(x):
                return floor((x / VALIDATION_IMG_CNT) * (10 ** LOSS_OUTPUT_PRECISION)) / (10 ** LOSS_OUTPUT_PRECISION)

            train_loss_stats.append(_to_output_t(sum_loss))
            train_loss_stats_noobj.append(_to_output_t(sum_loss_noobj))
            train_loss_stats_obj.append(_to_output_t(sum_loss_obj))
            train_loss_stats_cl.append(_to_output_t(sum_loss_cl))
            train_loss_stats_xy.append(_to_output_t(sum_loss_xy))
            train_loss_stats_wh.append(_to_output_t(sum_loss_wh))

            validation_loss_stats.append(_to_output_v(val_loss))
            validation_loss_stats_noobj.append(_to_output_v(val_loss_noobj))
            validation_loss_stats_obj.append(_to_output_v(val_loss_obj))
            validation_loss_stats_cl.append(_to_output_v(val_loss_cl))
            validation_loss_stats_xy.append(_to_output_v(val_loss_xy))
            validation_loss_stats_wh.append(_to_output_v(val_loss))

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

        for epoch in range(self.next_training_epoch, epochs, 1):

            try:
                
                new_lr = self.lr_scheduler(epoch, self.full_network.optimizer.learning_rate)
                self.full_network.optimizer.learning_rate = new_lr

                progbar_output = tf.keras.utils.Progbar(TRAIN_BATCH_CNT)
                tf.print(f"\nEpoch {epoch} (lr {new_lr}):")

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
                for (imgs, bool_mask_size1, target_mask_size1, bool_mask_size2, target_mask_size2, bool_mask_size3, target_mask_size3) in self.data_loader.load_data(TRAIN_BATCH_SIZE, "train"):

                    with tf.GradientTape() as tape:
                    
                        out_s1, out_s2, out_s3 = self.full_network(imgs, training=True)

                        loss_value, noobj, obj, cl, xy, wh = yolov3_loss_perscale(out_s1, bool_mask_size1, target_mask_size1)
                        '''print(f"total loss = {loss_value}")
                        print(f"no obj loss = {noobj}")
                        print(f"obj loss = {obj}")
                        print(f"classif loss = {cl}")
                        print(f"xy loss = {xy}")
                        print(f"wh loss = {wh}")
                        print("\n")'''

                        loss_value_, noobj_, obj_, cl_, xy_, wh_ = yolov3_loss_perscale(out_s2, bool_mask_size2, target_mask_size2)
                        '''print(f"total loss = {loss_value_}")
                        print(f"no obj loss = {noobj_}")
                        print(f"obj loss = {obj_}")
                        print(f"classif loss = {cl_}")
                        print(f"xy loss = {xy_}")
                        print(f"wh loss = {wh_}")
                        print("\n")'''                        
                        
                        loss_value__, noobj__, obj__, cl__, xy__, wh__ = yolov3_loss_perscale(out_s3, bool_mask_size3, target_mask_size3)
                        '''print(f"total loss = {loss_value_}")
                        print(f"no obj loss = {noobj_}")
                        print(f"obj loss = {obj_}")
                        print(f"classif loss = {cl_}")
                        print(f"xy loss = {xy_}")
                        print(f"wh loss = {wh_}")
                        print("\n")'''

                        loss_value += loss_value_ + loss_value__
                        noobj += noobj_ + noobj__
                        obj += obj_ + obj__
                        cl += cl_ + cl__
                        xy += xy_ + xy__
                        wh += wh_ + wh__

                    gradients = tape.gradient(loss_value, self.full_network.trainable_weights)
                    self.full_network.optimizer.apply_gradients(zip(gradients, self.full_network.trainable_weights))

                    batch_idx += 1
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

                for (imgs, bool_mask_size1, target_mask_size1, bool_mask_size2, target_mask_size2, bool_mask_size3, target_mask_size3) in self.data_loader.load_data(VALIDATION_BATCH_SIZE, "validation"):
                    
                    out_s1, out_s2, out_s3 = self.full_network(imgs, training=False)

                    loss_value_, noobj_, obj_, cl_, xy_, wh_ = yolov3_loss_perscale(out_s1, bool_mask_size1, target_mask_size1)
                    val_loss += loss_value_
                    val_loss_noobj += noobj_
                    val_loss_obj += obj_
                    val_loss_cl += cl_
                    val_loss_xy += xy_
                    val_loss_wh += wh_

                    loss_value_, noobj_, obj_, cl_, xy_, wh_ = yolov3_loss_perscale(out_s2, bool_mask_size2, target_mask_size2)
                    val_loss += loss_value_
                    val_loss_noobj += noobj_
                    val_loss_obj += obj_
                    val_loss_cl += cl_
                    val_loss_xy += xy_
                    val_loss_wh += wh_
                    
                    loss_value_, noobj_, obj_, cl_, xy_, wh_ = yolov3_loss_perscale(out_s3, bool_mask_size3, target_mask_size3)
                    val_loss += loss_value_
                    val_loss_noobj += noobj_
                    val_loss_obj += obj_
                    val_loss_cl += cl_
                    val_loss_xy += xy_
                    val_loss_wh += wh_

            except KeyboardInterrupt:

                if self.cache_key is not None:
                    tf.print(f"Training paused for model with key {self.cache_key} (idx {self.cache_idx}) at epoch {epoch}")

                    self._save_model(epoch)

                    # NOTE 
                    # if the interrupt is generated in the middle of _log_show_losses (which is not atomic),
                    # stats may become broken, so we make sure this does not happen

                    stats = [train_loss_stats, train_loss_stats_noobj, train_loss_stats_obj,
                            train_loss_stats_cl, train_loss_stats_xy, train_loss_stats_wh,
                            validation_loss_stats, validation_loss_stats_noobj, validation_loss_stats_obj,
                            validation_loss_stats_cl, validation_loss_stats_xy, validation_loss_stats_wh]

                    self._save_train_stats(stats)

                else:
                    tf.print("Training interrupted; there is no cache key so the intermediary model will not be saved.")

                return

            _log_show_losses()

            if self.cache_key is not None and checkpoint_sched(epoch, sum_loss, val_loss):

                self._save_model(epoch + 1)

                stats = [train_loss_stats, train_loss_stats_noobj, train_loss_stats_obj,
                            train_loss_stats_cl, train_loss_stats_xy, train_loss_stats_wh,
                            validation_loss_stats, validation_loss_stats_noobj, validation_loss_stats_obj,
                            validation_loss_stats_cl, validation_loss_stats_xy, validation_loss_stats_wh]

                self._save_train_stats(stats)

        if self.next_training_epoch >= epochs:
            tf.print(f"Model with key {self.cache_key} (idx {self.cache_idx}) is already trained (at least) {epochs} epochs.")

        else:
            tf.print(f"\nTraining for model with key {self.cache_key} (idx {self.cache_idx}) is done ({epochs} epochs).")

            if self.cache_key is not None:
                self._save_model(epochs)

            stats = [train_loss_stats, train_loss_stats_noobj, train_loss_stats_obj,
                    train_loss_stats_cl, train_loss_stats_xy, train_loss_stats_wh,
                    validation_loss_stats, validation_loss_stats_noobj, validation_loss_stats_obj,
                    validation_loss_stats_cl, validation_loss_stats_xy, validation_loss_stats_wh]

            self._save_train_stats(stats)

    def show_architecture_stats(self):

        self.full_network.summary()
        tf.keras.utils.plot_model(self.full_network, show_shapes=True)

    def predict(self, threshold=0.6):
        
        if self.full_network is None:
            tf.print("Network not yet initialized")
            return

        for (img, _, _, _, _, _, _) in self.data_loader.load_data(1, "validation"):

            out_scale1, out_scale2, out_scale3 = self.full_network(img, training=False)

            anchors_relative = [tf.cast(GRID_CELL_CNT[d] * (self.data_loader.anchors[d] / IMG_SIZE[0]), dtype=tf.float32) for d in range(SCALE_CNT)]
        
            output_xy_min_scale0, output_xy_max_scale0, output_class_scale0, output_class_maxp_scale0 = make_prediction_perscale(out_scale1, anchors_relative[0], threshold)
            output_xy_min_scale1, output_xy_max_scale1, output_class_scale1, output_class_maxp_scale1 = make_prediction_perscale(out_scale2, anchors_relative[1], threshold)
            output_xy_min_scale2, output_xy_max_scale2, output_class_scale2, output_class_maxp_scale2 = make_prediction_perscale(out_scale3, anchors_relative[2], threshold)

            output_xy_min = [output_xy_min_scale0, output_xy_min_scale1, output_xy_min_scale2]
            output_xy_max = [output_xy_max_scale0, output_xy_max_scale1, output_xy_max_scale2]
            output_class = [output_class_scale0, output_class_scale1, output_class_scale2]
            output_class_maxp = [output_class_maxp_scale0, output_class_maxp_scale1, output_class_maxp_scale2]

            show_prediction(np.array(img[0]), output_xy_min, output_xy_max, output_class, output_class_maxp, self.data_loader.onehot_to_name)
