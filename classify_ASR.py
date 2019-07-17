# -*- coding:utf-8 -*-

import re
from classification_engine import ClassificationEngine
from Config import Config
import os
import numpy as np
import sqlite3
import time

tf_model = "ASR_Model_edgetpu.tflite"
tf_model_dir = "tflite_model"

valid_data_path = os.path.join(
    os.getcwd(),
    "tflite_model",
    "validdata",
)


def main():

    model = os.path.join(os.getcwd(), tf_model_dir, tf_model)

    print("[classify_ASR] model：{}".format(model))

    # Initialize engine.
    engine = ClassificationEngine(model)

    ''' 取得驗證資料集 '''
    valid_data_name = ""
    valid_data = list()

    for read_dir in os.listdir(valid_data_path):

        # print("[Model_to_TFLite] Valid Directory：{}".format(read_dir))

        if os.path.isdir(os.path.join(valid_data_path, read_dir)):

            valid_data_name = read_dir

            print("[classify_ASR] Valid Name：{}".format(valid_data_name))

            for read_file in os.listdir(os.path.join(valid_data_path, read_dir)):

                print("[classify_ASR] Valid File：{}".format(read_file))

                with open(os.path.join(valid_data_path, read_dir, read_file), "r") as read:
                    temp_read_content = read.readlines()

                for read_row in temp_read_content:

                    # 將每一行先做去除尾巴換行符號，並進行資料切割
                    read_row = str(read_row).rstrip(" \n").split(" ")

                    # temp_list = list()

                    # 讀取已完成去除尾巴換行符號與進行資料切割之每列內之欄位資料
                    for feature in read_row:

                        # 存放每一列的欄位資料
                        valid_data.append(float(feature))

    valid_data = np.array(valid_data).astype('uint8')
    # print("[classify_ASR] Valid data：{}".format(valid_data))

    get_score_list = engine.ClassifyWithASR_Feature(valid_data, top_k=3)
    print("[classify_ASR] get_score_list：{}".format(get_score_list))

    dbconn = sqlite3.connect(
        os.path.join(
            os.getcwd(),
            Config.SQLite_DB_DirectoryName,
            Config.SQLite_name
        )
    )

    curs = dbconn.cursor()

    for result in get_score_list:

        print('-----------------------------------------------------------------')

        ''' 查詢預測出來分類編號對應分類名稱 '''
        SQL_select_syntax = '''
        SELECT {} FROM {} WHERE {} = '{}'
        '''.format(
            Config.column_Classname,  # 欄位名稱-ClassName
            Config.db_TableName,  # 查詢資料表
            Config.column_ClassNum,  # 欄位名稱-ClassNum
            result[0]  # 預測分類編號結果
        )

        SQL_run = curs.execute(SQL_select_syntax)
        SQL_result = curs.fetchall()

        print("[classify_ASR] Label：{} , Score：{}".format(
            SQL_result[0][0], result[1]
        ))

        print('-----------------------------------------------------------------')


if __name__ == '__main__':

    Start_Time = time.time()

    main()

    print("\nSpeed time: {:.2f}s\n".format((time.time() - Start_Time)))
