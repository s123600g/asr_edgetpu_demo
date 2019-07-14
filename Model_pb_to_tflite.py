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

需注意！！！
'lite.TFLiteConverter.from_keras_model_file()' 要求tensorflow版本最低為 1.12
針對 1.9-1.11 版本 使用 'lite.TocoConverter'
針對 1.7-1.8 版本 使用 'lite.toco_convert'
'''

from Config import Config

import argparse
import tensorflow as tf
import os
import time

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

    # try:

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

    ''' 讀取來源模型，針對使用標準Tensorflow版本 '''
    # converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(
    #     Config.Model_PB_Path, # 模型來源
    #     Config.input_arrays, # 來自模型輸入張量之凍結圖層清單，實際上是輸入層之名稱
    #     Config.output_arrays # 來自模型輸出張量之凍結圖層清單，實際上是輸出層之名稱
    # )

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
    ''' 設置轉換型態為 uint8 針對使用 tf_nightly '''
    converter.inference_type = tf.uint8  # for tf_nightly

    ''' 設置轉換型態為 uint8 針對使用標準Tensorflow版本 '''
    # converter.inference_type = tf.contrib.lite.constants.QUANTIZED_UINT8

    ''' 
    取得讀取模型之輸入層，回傳模型之輸入向量之名稱
    Returns a list of the names of the input tensors.
    https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter#get_input_arrays
    '''
    input_arrays = converter.get_input_arrays()

    # print("input_arrays: {}".format(input_arrays))

    converter.quantized_input_stats = {
        input_arrays[0]: (0.0, 255.0)}  # (mean, stddev)

    ''' 
    進行轉換模型為TFLite格式模型 
    https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter#convert
    '''
    tflite_model = converter.convert()

    ''' 輸出轉換後tflite格式模型 '''
    with open(output_model_path, "wb") as f:
        f.write(tflite_model)

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

    print("[Model_to_TFLite] '{}' --> output_details\n{}".format(
        str(output_model_path).split('/')[-1],
        output_details[0],
    ))

    # output_data = interpreter.get_tensor(output_details[0]['index'])
    # print(output_data)

    print("\nSpeed time: {:.2f}s\n".format((time.time() - Start_Time)))

    print("可在終端機使用下列命令，來進行EdgeTPU Compiler TFLite model\n>> edgetpu_compiler -s {} <<\n".format(
        output_model_path
    ))

    # except FileNotFoundError as err:

    #     print("\n>>> {} <<<".format(err))

    # except Exception as err:

    #     print("\n>>> {} <<<".format(err))
