# -*- coding:utf-8 -*-

import tensorflow as tf

class ASR_Model():

    def __init__(self, class_num):
        ''''----------------------------------------------------------------------------------------------------------------------------------'''

        self.Conv2D_1 = tf.keras.layers.Conv2D(
            32, kernel_size=(2, 2),
            padding='same',  activation='relu'
        )

        self.batch_norm_1 = tf.keras.layers.BatchNormalization(
            fused=False)

        self.MaxPooling2D_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.Conv2D_2 = tf.keras.layers.Conv2D(
            64,  kernel_size=(2, 2),
            padding='same',  activation='relu'
        )

        self.batch_norm_2 = tf.keras.layers.BatchNormalization(fused=False)

        self.MaxPooling2D_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.Conv2D_3 = tf.keras.layers.Conv2D(
            128,  kernel_size=(1, 1), padding='same'
        )

        self.Conv2D_4 = tf.keras.layers.Conv2D(
            256,  kernel_size=(1, 1), padding='same'
        )

        self.dense_1 = tf.keras.layers.Dense(1024)

        self.flatten = tf.keras.layers.Flatten()

        self.ouput_layer = tf.keras.layers.Dense(
            class_num,  activation='softmax')

    def __gen_nn_structure(self, inputs):

        self.Conv2D_1 = self.Conv2D_1(inputs)
        self.batch_norm_1 = self.batch_norm_1(self.Conv2D_1)
        self.MaxPooling2D_1 = self.MaxPooling2D_1(self.batch_norm_1)
        self.Conv2D_2 = self.Conv2D_2(self.MaxPooling2D_1)
        self.batch_norm_2 = self.batch_norm_2(self.Conv2D_2)
        # self.MaxPooling2D_2 = self.MaxPooling2D_2(self.batch_norm_2)
        self.Conv2D_3 = self.Conv2D_3(self.batch_norm_2)
        self.l2_norm = tf.math.l2_normalize(
            self.Conv2D_3,
            axis=-1,
            epsilon=1e-12,
            name='l2norm',
        )
        self.Conv2D_4 = self.Conv2D_4(self.l2_norm)
        self.flatten = self.flatten(self.Conv2D_4)

        last_layer = self.ouput_layer(self.flatten)

        return tf.keras.Model(inputs=inputs, outputs=last_layer)

    def call(self, inputs):

        x = self.__gen_nn_structure(inputs=inputs)

        return x
