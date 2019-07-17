# -*- coding:utf-8 -*-

'''
MIT License

Copyright (c) 2019 李俊諭 JYUN-YU LI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''


from Config import Config
from Gen_DataSet import Gen_DataSet

from tensorflow.python.client import device_lib
from tensorflow import keras
from tensorflow.keras import backend as k
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, BatchNormalization

import os
import tensorflow as tf
import time
import numpy as np
import history_plot


'''模型訓練參數配置-CNN'''
batch_size = 100
epochs = 20
quant_delay = 10
verbose = 1
# CNN_optimizer = 'Adadelta'
CNN_optimizer = keras.optimizers.Adadelta(lr=0.5)
# CNN_loss = 'categorical_crossentropy'
CNN_loss = keras.losses.categorical_crossentropy

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
CNN_twolayer_Dropout = 0.25

CNN_full_connectionlayer_Dense = 128
CNN_full_connectionlayer_Activation = 'relu'
CNN_full_connectionlayer_Dropout = 0.25
CNN_ouputlayer_Activation = 'softmax'

''' 設置模型訓練時之回調函數控制 '''
callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir=Config.Log_TensorBoard_Path,
        batch_size=batch_size,
        write_images=True,
    ),
    tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(Config.Model_ModelCheckpoint_Path, "ckpt_{epoch:02d}"),
                                       verbose=1,
                                       save_weights_only=False
                                       ),

]

input_arrays = list()
output_arrays = list()


def build_model():
    '''
    建置產生CNN模型實體。
    '''

    # 輸入層維度from keras.models import load_model
    input_shape = (
        np.array(Config.Train_DataSet).shape[1],
        np.array(Config.Train_DataSet).shape[2],
        Config.channel
    )

    # print("[Train_Data] input_shape： {}".format(
    #     input_shape
    # ))

    model = Sequential()

    ''' 建置輸入層-Input Layer '''
    # 輸入層
    model.add(
        Conv2D(
            CNN_inputlayer_conv2D_hidden_unit,
            kernel_size=CNN_inputlayer_conv2D_kernel_size,
            padding=CNN_inputlayer_conv2D_padding,
            input_shape=input_shape,
            # activation=CNN_inputlayer_Activation
        )
    )

    # 激活函數
    model.add(

        Activation(CNN_inputlayer_Activation)
    )

    ''' 建置第一層 Conv2D Layer '''
    model.add(
        Conv2D(
            CNN_onelayer_conv2D_hidden_unit,
            kernel_size=CNN_onelayer_conv2D_kernel_size,
            padding=CNN_onelayer_conv2D_padding,
        )
    )

    model.add(

        Activation(CNN_onelayer_Activation)
    )

    # 批量標準化層
    model.add(BatchNormalization(fused=False))

    # 池化層
    model.add(
        MaxPooling2D(
            pool_size=CNN_onelayer_MaxPooling2D_pool_size
        )
    )

    ''' 建置第二層 Conv2D Layer '''
    model.add(
        Conv2D(
            CNN_twolayer_conv2D_hidden_unit,
            kernel_size=CNN_twolayer_conv2D_kernel_size,
            padding=CNN_twolayer_conv2D_padding,
        )
    )

    # 激活函數
    model.add(

        Activation(CNN_twolayer_Activation)
    )

    # 批量標準化層
    model.add(BatchNormalization(fused=False))

    # 池化層
    model.add(
        MaxPooling2D(
            pool_size=CNN_twolayer_MaxPooling2D_pool_size
        )
    )

    ''' 建置全連接層、輸出層'''
    # 平坦層
    model.add(
        Flatten()
    )

    # 全連接層
    model.add(
        Dense(
            CNN_full_connectionlayer_Dense,
        )
    )

    # 激活函數
    model.add(

        Activation(CNN_full_connectionlayer_Activation)
    )

    # 輸出層
    model.add(
        Dense(
            Config.class_num,
            activation=CNN_ouputlayer_Activation
        )
    )

    return model


if __name__ == "__main__":

    # try:

    Start_Time = time.time()

    # print(device_lib.list_local_devices(), end="\n\n")

    print("[Train_Data] Tensorflow Version：{}".format(
        tf.version.VERSION))  # for tf_nightly

    # print("[Train_Data] Tensorflow Version：{}".format(tf.VERSION))

    print("[Train_Data] Tensorflow-Keras Version：{}".format(tf.keras.__version__))
    print("[Train_Data] CheckPoints Path：{}".format(
        Config.Model_checkpoints_Path
    ))

    ''' 產生訓練、測試、驗證資料集 '''
    Gen_DataSet = Gen_DataSet(Config)
    Gen_DataSet.DataSet_Process()

    ''' 分配每一回合訓練資料量 '''
    steps_per_epoch = int((len(Config.Train_Labels) / batch_size))

    print("steps_per_epoch：{}".format(
        int((len(Config.Train_Labels) / batch_size))
    ))

    ''' 
    tf.Graph()
    建立一個空的模型圖層 
    https://www.tensorflow.org/api_docs/python/tf/Graph#__init__
    '''
    train_graph = tf.Graph()
    train_sess = tf.compat.v1.Session(graph=train_graph)  # for tf_nightly
    # train_sess = tf.Session(graph=train_graph)

    tf.compat.v1.keras.backend.set_session(train_sess)  # for tf_nightly
    # tf.keras.backend.set_session(train_sess)

    with train_graph.as_default():

        ''' 
        設置學習階段執行動作，給予整數設置此處設為1，代表進行Train 
        https://keras.io/backend/
        '''
        tf.keras.backend.set_learning_phase(1)

        ''' 建置模型架構實體 '''
        net_model = build_model()

        ''' 輸出顯示模型架構總體資訊 '''
        net_model.summary()

        ''' 
        建立量化之訓練圖層，將輸入圖層重新建置模擬量化
        需要注意此步驟必須在執行梯度優化(optimizer)之前，也就是要在模型建置前向與後向之間，做這一步插入量化圖層動作。
        https://www.tensorflow.org/api_docs/python/tf/contrib/quantize/create_training_graph
        '''
        tf.contrib.quantize.create_training_graph(
            input_graph=train_graph,  # 模型圖層
            quant_delay=quant_delay  # 設置訓練期間每多少回合，進行權重與激活函數兩者進行量化
        )

        '''
        進行模型圖層之梯度優化全域變量初始化
        https://www.tensorflow.org/api_docs/python/tf/initializers/global_variables?authuser=0&hl=zh-Tw
        '''
        # for tf_nightly
        train_sess.run(tf.compat.v1.global_variables_initializer())
        # train_sess.run(tf.global_variables_initializer())

        ''' 編譯模型架構 '''
        net_model.compile(
            loss=CNN_loss,  # 損失函數
            optimizer=CNN_optimizer,  # 優化函數(針對梯度下降)
            metrics=['accuracy']
        )

        ''' 訓練模型 '''
        history = net_model.fit(
            # 訓練資料集
            Config.Train_DataSet,
            # 訓練資料集標籤
            Config.Train_Labels,
            # 設置每一回合訓練資料量
            batch_size=steps_per_epoch,
            # 設置訓練幾回合
            epochs=epochs,
            # 是否觀察訓練過程，設值為1代表要顯示觀察
            verbose=verbose,
            # 設置驗證資料集
            validation_data=(Config.Valid_DataSet, Config.Valid_Labels),
            # 回調函數
            callbacks=callbacks
        )

        ''' 將tensor圖層與紀錄點存檔 '''
        saver = tf.compat.v1.train.Saver()  # for tf_nightly
        # saver = tf.train.Saver()
        saver.save(train_sess, Config.Model_checkpoints_Path)

        ''' 建立訓練過程之準確度與損失函數變化圖片 '''
        history_plot.plot_figure(
            history,
            os.path.join(os.getcwd(), Config.Plot_Figure_DirectoryName),
            "Audio_Speech_Training"
        )

        ''' 驗證訓練後模型 '''
        score = net_model.evaluate(
            Config.Test_DataSet,
            Config.Test_Labels,
            verbose=verbose
        )

        print("\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(
            "Evalluate Loss：[{:.2f}] | Accuracy：[{:.2f}] ".format(score[0], score[1]))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        ''' 儲存訓練後模型和權重 '''
        net_model.save_weights(Config.Model_Weight_Path)
        net_model.save(Config.Model_Path)

    ''' Eval model '''
    eval_graph = tf.Graph()
    eval_sess = tf.compat.v1.Session(graph=eval_graph)  # for tf_nightly
    # tf.keras.backend.clear_session()
    # eval_sess = tf.Session(graph=eval_graph)
    tf.keras.backend.set_session(eval_sess)

    with eval_graph.as_default():

        ''' 
        設置學習階段執行動作，給予整數設置此處設為0，代表進行Test
        https://keras.io/backend/
        '''
        tf.keras.backend.set_learning_phase(0)

        ''' 取得模型圖層骨幹 '''
        eval_model = build_model()

        ''' 
        建立量化之驗證圖層，將輸入圖層重新建置模擬量化
        https://www.tensorflow.org/api_docs/python/tf/contrib/quantize/create_training_graph
        '''
        tf.contrib.quantize.create_eval_graph(
            input_graph=eval_graph  # 模型圖層
        )

        ''' 取得模型內部結構，包含節點、內部函數數值等等 '''
        eval_graph_def = eval_graph.as_graph_def()

        # print("eval_graph_def：\n{}\n".format(
        #     eval_graph_def
        # ))

        ''' 重新載入儲存權重變數，透過儲存紀錄點(存放權重參數) '''
        saver = tf.compat.v1.train.Saver()  # for tf_nightly
        # saver = tf.train.Saver()
        saver.restore(eval_sess, Config.Model_checkpoints_Path)

        ''' 
        將模型中所有的圖層變量轉換成相同數值常量
        https://www.tensorflow.org/api_docs/python/tf/graph_util/convert_variables_to_constants
        '''
        # for tf_nightly
        frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            eval_sess,
            eval_graph_def,
            [eval_model.output.op.name]
        )

        # frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        #     eval_sess,
        #     eval_graph_def,
        #     [eval_model.output.op.name]
        # )

        # print("[Train_Data]\n{}\n".format(
        #     [eval_model.output.op.name]
        # ))

        input_arrays.append(eval_model.input.op.name)
        print("\n[Train_Data] Model_Input_Layer_Name：{}".format(
            input_arrays[0]
        ))

        output_arrays.append(eval_model.output.op.name)
        print("[Train_Data] Model_Output_Layer_Name：{}".format(
            output_arrays[0]
        ))

        print("[Train_Data] Model_PB_Path：{}".format(Config.Model_PB_Path))

        ''' 輸出pb格式模型 '''
        with open(Config.Model_PB_Path, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())

    # # converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(
    # #     Config.Model_PB_Path,
    # #     input_arrays,
    # #     output_arrays
    # # )

    # # # for tf_nightly
    # converter = tf.lite.TFLiteConverter.from_frozen_graph(
    #     Config.Model_PB_Path,
    #     input_arrays,
    #     output_arrays
    # )

    # ''' 設置轉換型態為uint8/QUANTIZED_UINT8 '''
    # converter.inference_type = tf.uint8  # for tf_nightly
    # # converter.inference_type = tf.contrib.lite.constants.QUANTIZED_UINT8
    # input_arrays = converter.get_input_arrays()

    # # print("input_arrays: {}".format(input_arrays))
    # converter.quantized_input_stats = {
    #     input_arrays[0]: (0.0, 255.0)}  # (mean, stddev)

    # tflite_model = converter.convert()

    # ''' 輸出轉換後tflite格式模型 '''
    # with open(Config.Output_Model_Path, "wb") as f:
    #     f.write(tflite_model)

    end_time = '{:.2f}'.format((time.time() - Start_Time))

    print("\nSpeed time: {}s".format(end_time))

    # except Exception as err:

    #     print("\n>>> {} <<<".format(err))
