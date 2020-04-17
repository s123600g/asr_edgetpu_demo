# -*- coding:utf-8 -*-

import tensorflow as tf

slim = tf.contrib.slim


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

        # self.full_connectionlayer = tf.keras.layers.Dense(
        #     128, activation='relu')

        self.ouput_layer = tf.keras.layers.Dense(
            class_num,  activation='softmax')

    # def __gen_net_structure(self, inputs):

    #     with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.truncated_normal_initializer(stddev=0.01)):
    #         net = slim.conv2d(inputs, 8, [2, 2], scope='net1')

    #         net = slim.repeat(net, 2, slim.conv2d, 16, [2, 2], scope='net2')
    #         net = slim.batch_norm(net,  scope='net2_bn')
    #         net = slim.max_pool2d(net, [2, 2], scope='net2_maxpool')

    #         net = slim.repeat(net, 2, slim.conv2d, 32, [2, 2], scope='net3')
    #         net = slim.batch_norm(net,  scope='net3_bn')
    #         net = slim.max_pool2d(net, [2, 2], scope='net3_maxpool')

    #         net = slim.repeat(net, 2, slim.conv2d, 64, [2, 2], scope='net4')
    #         net = slim.batch_norm(net,  scope='net4_bn')
    #         net = slim.max_pool2d(net, [2, 2], scope='net4_maxpool')

    #     net = slim.conv2d(net, 128, [1, 1], scope='net5')
    #     net = tf.nn.l2_normalize(net, axis=-1, epsilon=1e-12, name='l2norm')

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
        # self.full_connectionlayer = self.full_connectionlayer(
        #     self.MaxPooling2D_2)
        last_layer = self.ouput_layer(self.flatten)

        return tf.keras.Model(inputs=inputs, outputs=last_layer)

    def call(self, inputs):

        x = self.__gen_nn_structure(inputs=inputs)

        return x
