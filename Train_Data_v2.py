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

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras import Input
from tensorflow.keras.backend import l2_normalize
from tensorflow.keras.utils import plot_model
from ASR_Model import ASR_Model
from Config import Config
from Gen_DataSet import Gen_DataSet
import os
import time
import numpy as np
import history_plot

'''模型訓練參數配置-CNN'''
batch_size = 1000
epochs = 2
quant_delay = 10
verbose = 1

''' 設置模型訓練時之回調函數控制 '''
callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir=Config.Log_TensorBoard_Path,
        batch_size=batch_size,
        write_images=True,
    ),
    tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(Config.Model_ModelCheckpoint_Path, "ckpt_{epoch:02d}"),
                                       verbose=1,
                                       save_weights_only=True
                                       ),
]

input_arrays = list()
output_arrays = list()

if __name__ == "__main__":

    Start_Time = time.time()
    print("[Train_Data_v2] device_lib：{}".format(
        device_lib.list_local_devices()))
    print("[Train_Data_v2] Tensorflow Version：{}".format(
        tf.version.VERSION))  # for tf_nightly

    print("[Train_Data_v2] Tensorflow-Keras Version：{}".format(tf.keras.__version__))
    print("[Train_Data_v2] CheckPoints Path：{}".format(
        Config.Model_checkpoints_Path
    ))

    ''' 產生訓練、測試、驗證資料集 '''
    Gen_DataSet = Gen_DataSet(Config)
    Gen_DataSet.DataSet_Process()

    # 輸入層維度from keras.models import load_model
    input_shape = (
        np.array(Config.Train_DataSet).shape[1],
        np.array(Config.Train_DataSet).shape[2],
        Config.channel
    )

    ''' 分配每一回合訓練資料量 '''
    steps_per_epoch = int((len(Config.Train_Labels) / batch_size))
    print("[Train_Data_v2]steps_per_epoch：{}".format(
        int((len(Config.Train_Labels) / batch_size))
    ))

    ''' 
    tf.Graph()
    建立一個空的模型圖層 
    https://www.tensorflow.org/api_docs/python/tf/Graph#__init__
    '''
    train_graph = tf.Graph()
    train_sess = tf.compat.v1.Session(graph=train_graph)  # for tf_nightly
    tf.compat.v1.keras.backend.set_session(train_sess)  # for tf_nightly

    with train_graph.as_default():
        ''' 
        設置學習階段執行動作，給予整數設置此處設為1，代表進行Train 
        https://keras.io/backend/
        '''
        tf.keras.backend.set_learning_phase(1)

        get_DataSet_dtype = Config.Train_DataSet.dtype
        gen_input_tensor = tf.convert_to_tensor(
            Config.Train_DataSet, dtype=get_DataSet_dtype)
        print(f"[Train_Data_v2] gen_input_tensor : {gen_input_tensor}")

        ''' 建置模型架構實體 '''
        net_model = ASR_Model(Config.class_num).call(
            inputs=tf.keras.Input(shape=input_shape, tensor=gen_input_tensor, dtype=get_DataSet_dtype))

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

        ''' 編譯模型架構 '''
        net_model.compile(
            loss=tf.keras.losses.categorical_crossentropy,  # 損失函數
            optimizer=tf.keras.optimizers.Adadelta(lr=0.5),  # 優化函數(針對梯度下降)
            metrics=['accuracy']
        )

        ''' 訓練模型 '''
        history = net_model.fit(
            # # 訓練資料集標籤
            y=Config.Train_Labels,
            # 設置每一回合訓練資料量
            # batch_size=steps_per_epoch,
            steps_per_epoch=steps_per_epoch,
            # 設置訓練幾回合
            epochs=epochs,
            # 是否觀察訓練過程
            verbose=verbose,
            # 回調函數
            callbacks=callbacks
        )

        ''' 將模型結構輸出成圖片檔 '''
        plot_model(
            net_model,
            to_file=os.path.join(
                os.getcwd(),
                'model',
                'model_visualized.png'
            )
        )

        ''' 將tensor圖層與紀錄點存檔 '''
        saver = tf.compat.v1.train.Saver()  # for tf_nightly
        saver.save(train_sess, Config.Model_checkpoints_Path)

        # 建立資料Tensor
        gen_test_tensor = tf.convert_to_tensor(
            Config.Test_DataSet, dtype=get_DataSet_dtype)
        print(f"[Train_Data_v2] gen_test_tensor : {gen_test_tensor}")

        ''' 驗證訓練後模型 '''
        score = net_model.evaluate(
            x=gen_test_tensor,
            y=Config.Train_Labels,
            steps=steps_per_epoch,
            verbose=0
        )

        print("\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("Evalluate Loss：[{:.2f}] | Accuracy：[{:.2f}] ".format(
            score[0], score[1]))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    ''' Eval model '''
    eval_graph = tf.Graph()
    eval_sess = tf.compat.v1.Session(graph=eval_graph)  # for tf_nightly
    tf.keras.backend.set_session(eval_sess)

    with eval_graph.as_default():

        ''' 
        設置學習階段執行動作，給予整數設置此處設為0，代表進行Test
        https://keras.io/backend/
        '''
        tf.keras.backend.set_learning_phase(0)

        ''' 建置模型架構實體 '''
        eval_model = ASR_Model(Config.class_num).call(
            inputs=tf.keras.Input(shape=input_shape))

        ''' 
        建立量化之驗證圖層，將輸入圖層重新建置模擬量化
        https://www.tensorflow.org/api_docs/python/tf/contrib/quantize/create_eval_graph
        '''
        tf.contrib.quantize.create_eval_graph(
            input_graph=eval_graph  # 模型圖層
        )

        ''' 取得模型內部結構，包含節點、內部函數數值等等 '''
        eval_graph_def = eval_graph.as_graph_def()

        # print("[Train_Data_v2]eval_graph_def：{}\n".format(
        #     eval_graph_def
        # ))

        print("[Train_Data_v2] trainable_variables：{}".format(
            tf.trainable_variables()))

        ''' 重新載入儲存權重變數，透過儲存紀錄點(存放權重參數) '''
        saver = tf.compat.v1.train.Saver()  # for tf_nightly
        saver.restore(eval_sess, Config.Model_checkpoints_Path)

        ''' 
        將模型中所有的圖層變量轉換成相同數值常量
        https://www.tensorflow.org/api_docs/python/tf/graph_util/convert_variables_to_constants
        '''
        # for tf_nightly
        frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            eval_sess,
            eval_graph_def,
            [eval_model.output.op.name],

        )

        print("[Train_Data_v2] output.op.name：{}\n".format(
            [eval_model.output.op.name]
        ))

        # input_arrays.append(eval_model.input.op.name)
        # print("\n[Train_Data_v2] Model_Input_Layer_Name：{}".format(
        #     input_arrays[0]
        # ))

        # output_arrays.append(eval_model.output.op.name)
        # print("[Train_Data_v2] Model_Output_Layer_Name：{}".format(
        #     output_arrays[0]
        # ))

        print("[Train_Data_v2] Model_PB_Path：{}".format(Config.Model_PB_Path))

        ''' 輸出pb格式模型 '''
        with open(Config.Model_PB_Path, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())

    end_time = '{:.2f}'.format((time.time() - Start_Time))

    print("\nSpeed time: {}s".format(end_time))
