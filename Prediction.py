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

# from keras.models import load_model
from tensorflow.keras.models import load_model
# from keras.applications.imagenet_utils import decode_predictions
from Config import Config

import os
import numpy as np
import sqlite3
import time
import librosa

dbconn = sqlite3.connect(
    os.path.join(
        os.getcwd(),
        Config.SQLite_DB_DirectoryName,
        Config.SQLite_name
    )
)

curs = dbconn.cursor()
batch_size = 126
correct_count = 0
err_count = 0
is_one_line = True


def gen_wav_mfcc(file_path, sample_rate, max_pad_len):
    '''
   產生語音Wav檔案之特徵值(MFCC)。\n

    所需參數如下：\n
    file_path: 語音檔案放置位置\n
    sample_rate：語音檔案之頻率\n
    max_pad_len: 語音特徵向量補值最大長度\n

    '''

    wave, sr = librosa.load(file_path, mono=True, sr=sample_rate)

    # print("[Prediction]wave type：{}".format(type(wave)))

    # print("wave shape：{}".format(wave.shape))

    # wave array begins at 0 and ends at array length with step 3
    wave = wave[::3]

    # print("wave shape：{}".format(wave.shape))
    # print("wave:\n{}\n".format(wave))

    # 取得語音檔案MFCC特徵值
    mfcc = librosa.feature.mfcc(wave, sr=sample_rate)

    # print("mfcc：\n{}\n".format(mfcc))
    # print("[Prediction]mfcc type：{}".format(type(mfcc)))
    # print("[Prediction]mfcc shape：{}\n".format(mfcc.shape))

    # 設置資料維度補齊長度
    pad_width = max_pad_len - mfcc.shape[1]

    # 針對資料維度補齊0
    mfcc = np.pad(mfcc, pad_width=(
        (0, 0), (0, pad_width)), mode='constant')

    # print("[Prediction]new mfcc type：{}".format(type(mfcc)))
    # print("[Prediction]new mfcc shape：{}\n".format(mfcc.shape))
    # print("[Prediction]\n{}".format(mfcc))

    # print()

    return mfcc


if __name__ == "__main__":

    start_time = time.time()

    print("\nWait for configure operation sources......", end="\n\n")

    print("[Prediction] Start load Model from [{}]\n".format(
        Config.Model_Path
    ), end="\n\n")

    ''' 載入模型 '''
    get_model = load_model(
        Config.Model_Path
    )

    ''' 顯示輸出讀取模型架構資訊 '''
    get_model.summary()

    ''' 判斷放置所有預測資料總目錄是否存在 '''
    if os.path.exists(Config.Prediction_Audio_Data_Path):

        ''' 判斷放置所有預測資料總目錄是否內部有資料存在 '''
        if len(os.listdir(Config.Prediction_Audio_Data_Path)) != 0:

            # 開啟輸出識別紀錄檔案
            filewrite = open(
                os.path.join(
                    os.getcwd(),
                    Config.Log_DirectoryName,
                    Config.Log_Recognition_Result_DirectoryName,
                    str(Config.Log_Recognition_Result_DirectoryName +
                        "." + Config.log_file_type)
                ),
                'w'
            )

            ''' 讀取放置所有預測資料總目錄內部 '''
            for read_item in os.listdir(Config.Prediction_Audio_Data_Path):

                ''' 判斷目前讀取到的項目是否為目錄 '''
                if os.path.isdir(os.path.join(Config.Prediction_Audio_Data_Path, read_item)):

                    print(
                        "\n[Prediction] Current read folder [ '{}' ]. Start recognition".format(read_item))

                    if is_one_line:

                        filewrite.write(
                            ">>> Folder name [ '{}' ] recognition result：\n".format(
                                read_item
                            )
                        )

                        is_one_line = False

                    else:

                        filewrite.write(
                            "\n>>> Folder name [ '{}' ] recognition result：\n".format(
                                read_item
                            )
                        )

                    temp_correct_count = 0
                    temp_err_count = 0

                    for read_file in os.listdir(os.path.join(Config.Prediction_Audio_Data_Path, read_item)):

                        # 擷取語音wav檔案MFCC特徵值
                        mfcc_data = gen_wav_mfcc(
                            os.path.join(
                                Config.Prediction_Audio_Data_Path,
                                read_item,
                                read_file
                            ),
                            Config.sample_rate,
                            Config.max_pad_len,
                        )

                        ''' 將音頻MFCC特徵矩陣轉換為numpy array '''
                        mfcc_data = np.array(mfcc_data)
                        ''' 重新設置音頻MFCC特徵值矩陣維度，格式為[row , column , single channel] '''
                        mfcc_data = mfcc_data.reshape(
                            mfcc_data.shape[0], mfcc_data.shape[1], Config.channel)
                        ''' 將音頻MFCC特徵值矩陣維度第一個位置新增補值，補值為代表為單個特徵數值矩陣，格式為[file quantity ,row , column , single channel] '''
                        mfcc_data = np.expand_dims(mfcc_data, axis=0)
                        ''' 將音頻MFCC特徵值矩陣小數精度轉換為float32 '''
                        mfcc_data = mfcc_data.astype('float32')

                        # print("[Prediction]mfcc_data shape：{}".format(
                        #     mfcc_data.shape))

                        ''' 進行語音預測 '''
                        predict_result = get_model.predict(
                            mfcc_data,
                            batch_size=batch_size
                        )

                        get_prediction_index = np.argmax(predict_result[0])

                        # print("{:.2f}%".format(
                        #     (predict_result[0][get_prediction_index] * 100)))
                        # print("{}".format(get_prediction_index))

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

                        ''' 判斷識別分類結果是否正確 '''
                        if SQL_result[0][0] == read_item:

                            print("(O) [ {} ] ---> [ '{}' ]({:.2f}%)".format(
                                read_file,
                                SQL_result[0][0],
                                (predict_result[0][get_prediction_index] * 100)
                            ))

                            filewrite.write("(O) [ {} ] ---> ['{}']({:.2f}%)\n".format(
                                read_file,
                                SQL_result[0][0],
                                (predict_result[0][get_prediction_index] * 100)
                            ))

                            correct_count += 1
                            temp_correct_count += 1

                        else:

                            print("(X) [ {} ] ---> [ '{}' ]({:.2f}%)".format(
                                read_file,
                                SQL_result[0][0],
                                (predict_result[0][get_prediction_index] * 100)
                            ))

                            filewrite.write("(X) [ {} ] ---> ['{}']({:.2f}%)\n".format(
                                read_file,
                                SQL_result[0][0],
                                (predict_result[0][get_prediction_index] * 100)
                            ))

                            err_count += 1
                            temp_err_count += 1

                    print('''----------------------------------------------\nThe label ['{}'] recognition result：\ncorrect   total: {} count.\nincorrect total: {} count.\n----------------------------------------------
                    '''.format(
                        read_item,
                        temp_correct_count,
                        temp_err_count
                    ))

                    filewrite.write(
                        '''----------------------------------------------\nThe label ['{}'] recognition result：\ncorrect   total: {} count.\nincorrect total: {} count.\n----------------------------------------------\n
                    '''.format(
                            read_item,
                            temp_correct_count,
                            temp_err_count
                        )
                    )

            end_time = '{:.2f}'.format((time.time() - start_time))

            filewrite.write(
                '''
                \n
                --------------------------------------
                Audio Prediction Data Quantity：{}
                correct   total: {} count.
                incorrect total: {} count.
                --------------------------------------
                Full Speed Time: {}s
                '''.format(
                    (correct_count + err_count),
                    correct_count,
                    err_count,
                    end_time
                )
            )

            # 關閉輸出紀錄檔案
            filewrite.close()

            print(
                '''
                \n
                --------------------------------------
                Audio Prediction Data Quantity：{}
                correct   total: {} count.
                incorrect total: {} count.
                --------------------------------------
                Full Speed Time: {}s 
                '''.format(
                    (correct_count + err_count),
                    correct_count,
                    err_count,
                    end_time
                )
            )

        else:

            print("[Prediction] The path [{}] is empty.".format(
                Config.Prediction_Audio_Data_Path
            ))

    else:

        print("[Prediction] The path [{}] can not found.".format(
            Config.Prediction_Audio_Data_Path
        ))
