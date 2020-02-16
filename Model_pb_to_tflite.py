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


Exporting a tf.keras File
https://www.tensorflow.org/lite/convert/python_api#exporting_a_tfkeras_file_

How to convert keras(h5) file to a tflite file?
https://stackoverflow.com/questions/53256877/how-to-convert-kerash5-file-to-a-tflite-file

'''

from Config import Config

import argparse
import tensorflow as tf
import os
import time
import numpy as np
import sqlite3

''' 設置CLI執行程式時，參數項目配置 '''
parser = argparse.ArgumentParser()

''' 建立--input參數 '''
parser.add_argument("--input", type=str,
                    help="Set a model sources path for input.", default="None")

''' 建立--output參數 '''
parser.add_argument("--output", type=str,
                    help="Set a save path for output tflite model", default="None")

args = parser.parse_args()

input_model_path = ""  # 放置來源模型位置
output_model_path = ""  # 放置輸出模型位置
valid_data_path = os.path.join(
    os.getcwd(),
    "tflite_model",
    "validdata",
)


def gen_Path_DetailInfo():

    global input_model_path, output_model_path

    if args.input == "None":

        print("[Model_to_TFLite] No set argument ['{}'] value, automated use default {} value.".format(
            "--input",
            "input"
        ))

        input_model_path = Config.Model_PB_Path

    else:

        input_model_path = args.input

    if args.output == "None":

        print("[Model_to_TFLite] No set argument ['{}'] value, automated use default {} value.".format(
            "--output",
            "output"
        ))

        output_model_path = Config.Output_Model_Path

    else:

        output_model_path = args.output

    return input_model_path, output_model_path


if __name__ == "__main__":

    Start_Time = time.time()

    ''' 產生輸入與輸出模型之路徑配置 '''
    gen_Path_DetailInfo()

    print("[Model_to_TFLite] Input path：[ {} ]".format(input_model_path))
    print("[Model_to_TFLite] Output path：[ {} ]".format(output_model_path))

    print()

    ''' 檢查模型來源是否不存在 '''
    if not os.path.exists(input_model_path):

        raise FileNotFoundError(
            "The ['{}'] can't found input model file.".format(
                input_model_path
            )
        )

    '''
    讀取來源模型，此來源模型為已凍結圖層之模型，針對使用 tf_nightly
    https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter#from_frozen_graph
    '''
    # for tf_nightly
    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        input_model_path,  # 模型來源
        Config.input_arrays,  # 來自模型輸入張量之凍結圖層清單，實際上是輸入層之名稱
        Config.output_arrays  # 來自模型輸出張量之凍結圖層清單，實際上是輸出層之名稱
    )

    '''
    Class TFLiteConverter
    https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter#attributes

    For 'inference_type' and 'inference_input_type' Attributes
    -----------------------------------------------------------------------------------------------------------------------------
    If inference_type is tf.uint8, signaling conversion to a fully quantized model from a quantization-aware trained input model,
    then inference_input_type defaults to tf.uint8
    如果converter.inference_type是tf.uint8，在信號轉換完整量化模型中，輸入模型經過量化感知訓練過程，預設inference_input_type為tf.uint8格式。

    For 'inference_output_type' and 'inference_input_type' Attributes
    -----------------------------------------------------------------------------------------------------------------------------
    If inference_type is tf.uint8, signaling conversion to a fully quantized model from a quantization-aware trained output model,
    then inference_output_type defaults to tf.uint8.
    如果converter.inference_type是tf.uint8，在信號轉換完整量化模型中，輸入模型經過量化感知訓練過程，預設inference_output_type為tf.uint8格式。

    '''
    ''' 設置權重型態轉換為 uint8 針對使用 tf_nightly '''
    converter.inference_type = tf.uint8 

    '''
    取得讀取模型之輸入層，回傳模型之輸入向量之名稱
    Returns a list of the names of the input tensors.
    https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter#get_input_arrays
    '''
    input_arrays = converter.get_input_arrays()

    # print("input_arrays: {}".format(input_arrays))

    converter.quantized_input_stats = {
        input_arrays[0]: (0.0, 255.0)  # (mean, stddev)
    }

    '''
    進行轉換模型為TFLite格式模型
    https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter#convert
    '''
    tflite_model = converter.convert()

    ''' 輸出轉換後tflite格式模型 '''
    with open(output_model_path, "wb") as f:
        f.write(tflite_model)

    convert_speed_time = (time.time() - Start_Time)

    print("\nConvert Speed Time: {:.2f}s\n".format(convert_speed_time))

    print("\n[Model_to_TFLite] 讀取量化轉換後TFLite model >> [{}] <<".format(
        output_model_path
    ))

    ''' 
    讀取轉換後 TFLite model and allocate tensors. 
    https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter?hl=zh-Tw&authuser=0
    '''
    interpreter = tf.lite.Interpreter(model_path=output_model_path)
    interpreter.allocate_tensors()

    ''' 取得 input and output tensors 資訊 '''
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\n[Model_to_TFLite] '{}' --> input_details：\n{}\n".format(
        str(output_model_path).split('/')[-1],
        input_details[0],
    ))

    print("[Model_to_TFLite] '{}' --> output_details\n{}\n".format(
        str(output_model_path).split('/')[-1],
        output_details[0],
    ))

    ''' 取得驗證資料集 '''
    valid_data_name = ""
    valid_data = list()

    for read_dir in os.listdir(valid_data_path):

        # print("[Model_to_TFLite] Valid Directory：{}".format(read_dir))

        if os.path.isdir(os.path.join(valid_data_path, read_dir)):

            valid_data_name = read_dir

            print("[Model_to_TFLite] Valid Name：{}".format(valid_data_name))

            for read_file in os.listdir(os.path.join(valid_data_path, read_dir)):

                print("[Model_to_TFLite] Valid File：{}".format(read_file))

                with open(os.path.join(valid_data_path, read_dir, read_file), "r") as read:
                    temp_read_content = read.readlines()

                for read_row in temp_read_content:

                    # 將每一行先做去除尾巴換行符號，並進行資料切割
                    read_row = str(read_row).rstrip(" \n").split(" ")

                    temp_list = list()

                    # 讀取已完成去除尾巴換行符號與進行資料切割之每列內之欄位資料
                    for feature in read_row:

                        # 存放每一列的欄位資料
                        valid_data.append(float(feature))

    valid_data = np.array(valid_data).astype('float32')
    # print("[Model_to_TFLite] Valid data：{}".format(valid_data))

    ''' 取得標準值與平均值 '''
    std, mean = input_details[0]['quantization']
    print("[Model_to_TFLite] std：{} , mean：{}".format(std, mean))

    ''' 將valid_data進行量化 '''
    quantize_valid_data = (valid_data / std + mean).astype('uint8')
    # print("[Model_to_TFLite] quantize_data：{}".format(
    #     (valid_data / std + mean)))
    # print("[Model_to_TFLite] quantize_valid_data：{}".format(quantize_valid_data))

    ''' 將valid_data進行reshape，跟input_details[0]['shape']一樣格式 '''
    sample_input_data = quantize_valid_data.reshape(input_details[0]['shape'])
    print("[Model_to_TFLite] sample_input_data shape：{}".format(
        sample_input_data.shape))

    ''' 設定輸入層張量Data '''
    interpreter.set_tensor(input_details[0]['index'], sample_input_data)

    ''' 進行模型調用 '''
    interpreter.invoke()

    predict_quantized_result = interpreter.get_tensor(
        output_details[0]['index']
    )

    print("[Model_to_TFLite] predict_quantized_result shape：{}".format(
        predict_quantized_result[0].shape
    ))

    print("[Model_to_TFLite] {} ".format(
        predict_quantized_result[0]
    ))

    get_prediction_index = np.argmax(predict_quantized_result[0])

    print("[Model_to_TFLite] get_prediction_index：{} ".format(
        get_prediction_index
    ))

    dbconn = sqlite3.connect(
        os.path.join(
            os.getcwd(),
            Config.SQLite_DB_DirectoryName,
            Config.SQLite_name
        )
    )

    curs = dbconn.cursor()

    ''' 查詢預測出來分類編號對應分類名稱 '''
    SQL_select_syntax = '''
    SELECT {} FROM {} WHERE {} = '{}'
    '''.format(
        Config.column_Classname,  # 欄位名稱-ClassName
        Config.db_TableName,  # 查詢資料表
        Config.column_ClassNum,  # 欄位名稱-ClassNum
        get_prediction_index  # 預測分類編號結果
    )

    SQL_run = curs.execute(SQL_select_syntax)
    SQL_result = curs.fetchall()

    print("[Model_to_TFLite] predict_quantized_result：{}".format(
        SQL_result[0][0],
    ))

    all_speed_time = (time.time() - Start_Time)

    print("\nSpeed time: {:.2f}s\n".format(all_speed_time))

    print("可在終端機使用下列命令，來進行EdgeTPU Compiler TFLite model\n>> edgetpu_compiler -s {} <<\n".format(
        output_model_path
    ))
