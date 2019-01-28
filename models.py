import tensorflow as tf
import os
import numpy as np
import cv2
import scipy.io as sio
import utils

import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

# np.set_printoptions(threshold=np.nan)

EPOCHS=25

class AlexNet:
    def __init__(self, dataset, class_num, batch_size, input_size):
        self.class_num = class_num
        self.batch_size = batch_size
        self.input_size = input_size
        self.idx_tensor = [idx for idx in range(self.class_num)]
        self.idx_tensor = tf.Variable(np.array(self.idx_tensor, dtype=np.float32))
        self.dataset = dataset
        self.model = self.__create_model()
        
    def __loss_angle(self, y_true, y_pred, alpha=0.5):
        # cross entropy loss
        bin_true = y_true[:,0]
        cont_true = y_true[:,1]
        cls_loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.keras.utils.to_categorical(bin_true, 66), logits=y_pred)
        # MSE loss
        pred_cont = tf.reduce_sum(tf.nn.softmax(y_pred) * self.idx_tensor, 1) * 3 - 99
        mse_loss = tf.losses.mean_squared_error(labels=cont_true, predictions=pred_cont)
        # Total loss
        total_loss = cls_loss + alpha * mse_loss
        return total_loss

    def __create_model(self):
        inputs = tf.keras.layers.Input(shape=(self.input_size, self.input_size, 3))
        
        feature = tf.keras.layers.Conv2D(filters=64, kernel_size=(11, 11), strides=4, padding='same', activation=tf.nn.relu)(inputs)
        feature = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(feature)
        feature = tf.keras.layers.Conv2D(filters=192, kernel_size=(5, 5), padding='same', activation=tf.nn.relu)(feature)
        feature = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(feature)
        feature = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)(feature)
        feature = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)(feature)
        feature = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)(feature)
        feature = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(feature)
        feature = tf.keras.layers.Flatten()(feature)
        feature = tf.keras.layers.Dropout(0.5)(feature)
        feature = tf.keras.layers.Dense(units=4096, activation=tf.nn.relu)(feature)
        
        fc_yaw = tf.keras.layers.Dense(name='yaw', units=self.class_num)(feature)
        fc_pitch = tf.keras.layers.Dense(name='pitch', units=self.class_num)(feature)
        fc_roll = tf.keras.layers.Dense(name='roll', units=self.class_num)(feature)
    
        model = tf.keras.Model(inputs=inputs, outputs=[fc_yaw, fc_pitch, fc_roll])
        
        losses = {
            'yaw':self.__loss_angle,
            'pitch':self.__loss_angle,
            'roll':self.__loss_angle,
        }
        
        model.compile(optimizer=tf.train.AdamOptimizer(),
                        loss=losses)
       
        return model

    def train(self, model_path, max_epoches=EPOCHS, load_weight=True):
        self.model.summary()
        
        if load_weight:
            self.model.load_weights(model_path)
        else:
            self.model.fit_generator(generator=self.dataset.data_generator(test=False),
                                    epochs=max_epoches,
                                    steps_per_epoch=self.dataset.train_num // self.batch_size,
                                    max_queue_size=10,
                                    workers=1,
                                    verbose=1)

            self.model.save(model_path)
            
    def test(self, save_dir):
        for i, (images, [batch_yaw, batch_pitch, batch_roll], names) in enumerate(self.dataset.data_generator(test=True)):
            predictions = self.model.predict(images, batch_size=self.batch_size, verbose=1)
            predictions = np.asarray(predictions)
            pred_cont_yaw = tf.reduce_sum(tf.nn.softmax(predictions[0,:,:]) * self.idx_tensor, 1) * 3 - 99
            pred_cont_pitch = tf.reduce_sum(tf.nn.softmax(predictions[1,:,:]) * self.idx_tensor, 1) * 3 - 99
            pred_cont_roll = tf.reduce_sum(tf.nn.softmax(predictions[2,:,:]) * self.idx_tensor, 1) * 3 - 99
            # print(pred_cont_yaw.shape)
            
            self.dataset.save_test(names[0], save_dir, [pred_cont_yaw[0], pred_cont_pitch[0], pred_cont_roll[0]])
            
    def test_online(self, face_imgs):
        batch_x = np.array(face_imgs, dtype=np.float32)
        predictions = self.model.predict(batch_x, batch_size=1, verbose=1)
        predictions = np.asarray(predictions)
        # print(predictions)
        pred_cont_yaw = tf.reduce_sum(tf.nn.softmax(predictions[0, :, :]) * self.idx_tensor, 1) * 3 - 99
        pred_cont_pitch = tf.reduce_sum(tf.nn.softmax(predictions[1, :, :]) * self.idx_tensor, 1) * 3 - 99
        pred_cont_roll = tf.reduce_sum(tf.nn.softmax(predictions[2, :, :]) * self.idx_tensor, 1) * 3 - 99
        
        return pred_cont_yaw[0], pred_cont_pitch[0], pred_cont_roll[0]
        