# -*- coding:utf-8 -*-

import tensorflow as tf

CNN_optimizer = tf.keras.optimizers.Adadelta(lr=0.5)
CNN_loss = tf.keras.losses.categorical_crossentropy

CNN_inputlayer_conv2D_hidden_unit = 32
CNN_inputlayer_conv2D_kernel_size = (2, 2)
CNN_inputlayer_Activation = 'relu'
CNN_inputlayer_conv2D_padding = 'same'

CNN_onelayer_conv2D_hidden_unit = 32
CNN_onelayer_conv2D_kernel_size = (2, 2)
CNN_onelayer_conv2D_padding = 'same'
CNN_onelayer_Activation = 'relu'
CNN_onelayer_MaxPooling2D_pool_size = (2, 2)

CNN_twolayer_conv2D_hidden_unit = 64
CNN_twolayer_conv2D_kernel_size = (2, 2)
CNN_twolayer_conv2D_padding = 'same'
CNN_twolayer_Activation = 'relu'
CNN_twolayer_MaxPooling2D_pool_size = (2, 2)

CNN_full_connectionlayer_Dense = 128
CNN_full_connectionlayer_Activation = 'relu'
CNN_ouputlayer_Activation = 'softmax'


class ASR_Model():

    def __init__(self, class_num):

        self.l2_norm = tf.keras.layers.Lambda(
            lambda x: tf.keras.backend.l2_normalize(x, axis=0))

        self.Conv2D_1 = tf.keras.layers.Conv2D(
            CNN_inputlayer_conv2D_hidden_unit,
            kernel_size=CNN_inputlayer_conv2D_kernel_size,
            padding=CNN_inputlayer_conv2D_padding,
        )

        self.activation_1 = tf.keras.layers.Activation(
            CNN_inputlayer_Activation)

        self.batch_norm_1 = tf.keras.layers.BatchNormalization(
            fused=False)

        self.MaxPooling2D_1 = tf.keras.layers.MaxPooling2D(
            pool_size=CNN_onelayer_MaxPooling2D_pool_size)

        self.Conv2D_2 = tf.keras.layers.Conv2D(
            CNN_onelayer_conv2D_hidden_unit,
            kernel_size=CNN_onelayer_conv2D_kernel_size,
            padding=CNN_onelayer_conv2D_padding,
        )

        self.activation_2 = tf.keras.layers.Activation(
            CNN_inputlayer_Activation)

        self.batch_norm_2 = tf.keras.layers.BatchNormalization(fused=False)

        self.MaxPooling2D = tf.keras.layers.MaxPooling2D(
            pool_size=CNN_onelayer_MaxPooling2D_pool_size)

        self.flatten = tf.keras.layers.Flatten()

        self.dense_1 = tf.keras.layers.Dense(CNN_full_connectionlayer_Dense)

        self.activation_3 = tf.keras.layers.Activation(
            CNN_full_connectionlayer_Activation)

        self.dense_2 = tf.keras.layers.Dense(
            class_num,  activation=CNN_ouputlayer_Activation)

    def __gen_nn_structure(self, inputs):

        self.l2_norm = self.l2_norm(inputs)
        self.Conv2D_1 = self.Conv2D_1(self.l2_norm)
        self.activation_1 = self.activation_1(self.Conv2D_1)
        self.batch_norm_1 = self.batch_norm_1(self.activation_1)
        self.MaxPooling2D_1 = self.MaxPooling2D_1(self.batch_norm_1)
        self.Conv2D_2 = self.Conv2D_2(self.MaxPooling2D_1)
        self.activation_2 = self.activation_2(self.Conv2D_2)
        self.batch_norm_2 = self.batch_norm_2(self.activation_2)
        self.MaxPooling2D = self.MaxPooling2D(self.batch_norm_2)
        self.flatten = self.flatten(self.MaxPooling2D)
        self.dense_1 = self.dense_1(self.flatten)
        self.activation_3 = self.activation_3(self.dense_1)
        last_layer = self.dense_2(self.activation_3)

        return tf.keras.Model(inputs=inputs, outputs=last_layer)

    def call(self, inputs):

        x = self.__gen_nn_structure(inputs=inputs)

        return x
