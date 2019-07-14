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

from scipy.io import wavfile

import librosa
import os
import numpy as np


class Load_Data():

    def __init__(self, Config):

        self.Config = Config

        self.class_labels = []

        print("[Load_Data]Current data floder name：{}".format(
            self.Config.Audio_Data_DirectoryName)
        )

        print("[Load_Data]Current data path：{}".format(
            self.Config.Audio_Data_Path), end="\n\n")

    def gen_wav_mfcc(self, file_path, sample_rate, max_pad_len):
        '''
       產生語音Wav檔案之特徵值(MFCC)。\n

        所需參數如下：\n
        file_path: 語音檔案放置位置\n
        sample_rate：語音檔案之頻率\n
        max_pad_len: 語音特徵向量補值最大長度\n

        '''

        wave, sr = librosa.load(file_path, mono=True, sr=sample_rate)

        # print("[Load_Data]wave type：{}".format(type(wave)))

        # print("wave shape：{}".format(wave.shape))

        # wave array begins at 0 and ends at array length with step 3
        wave = wave[::3]

        # print("wave shape：{}".format(wave.shape))
        # print("wave:\n{}\n".format(wave))

        # 取得語音檔案MFCC特徵值
        mfcc = librosa.feature.mfcc(wave, sr=sample_rate)

        # print("mfcc：\n{}\n".format(mfcc))
        # print("[Load_Data]mfcc type：{}".format(type(mfcc)))
        # print("[Load_Data]mfcc shape：{}\n".format(mfcc.shape))

        # 設置資料維度補齊長度
        pad_width = max_pad_len - mfcc.shape[1]

        # 針對資料維度補齊0
        mfcc = np.pad(mfcc, pad_width=(
            (0, 0), (0, pad_width)), mode='constant')

        # print("[Load_Data]new mfcc type：{}".format(type(mfcc)))
        # print("[Load_Data]new mfcc shape：{}\n".format(mfcc.shape))
        # print("[Load_Data]\n{}".format(mfcc))

        # print()

        return mfcc

    def __ouputlog_audio_feature(self, path, audio_feature):
        '''
        輸出音頻特徵值至對應檔案。\n

        關於參數：\n
        "path" --> 指定音頻特徵值輸出檔案放置路徑位置 \n
        "audio_feature" --> 指定音頻特徵值內容 \n

        有關音頻特徵值檔案輸出，參數配置在"Config.py"內輸出檔案參數配置區塊，參數資訊如下：\n
        "Log_FeatureData_DirectoryName" --> 放置每一分類特徵值檔案總目錄 \n
        '''

        with open(path, "w") as log_file:

            for read_row in audio_feature:

                for read_content in read_row:

                    log_file.write(str(read_content) + " ")

                log_file.write("\n")

    def __ouputlog_audio_class(self, path, audio_class):
        '''
        輸出音頻分類與對應編號至對應檔案。 \n

        關於參數：\n
        "path" --> 指定音頻特徵值輸出檔案放置路徑位置 \n
        "audio_class" --> 指定音頻所屬分類標籤資訊與編號內容 \n
        '''
        # print("Path：{}".format(path))

        with open(path, "w") as log_file:

            for index, value in audio_class:

                # print("NO.{} , Class：{}".format(index, value))

                log_file.write(str(index) + " " + str(value))

                log_file.write("\n")

    def __show_class_num_info(self, class_labels, class_labels_length):
        '''
        輸出顯示所有的分類標籤與其對應編號資訊。
        '''

        print("[Load_Data]\nAudio class：[ ", end="")

        for index, value in class_labels:

            if index != (class_labels_length - 1):

                print("{}：{}".format(index, value), end=" , ")

            else:

                print("{}：{}".format(index, value), end=" ")

        print("]\n")

    def load_data(self):
        '''
        讀取指定音頻目錄來源內，所有音頻檔案wav內容，並進行音頻特徵值與分類資訊擷取輸出特徵值檔案與分類資訊檔案。\n

        有關音頻來源相關設置，參數配置在"Config.py"內音頻資料來源參數設置區塊，參數資訊如下：\n
        "Audio_Data_DirectoryName" --> 放置每一分類音頻來源檔案(WAV)總目錄名稱 \n
        "Audio_Data_Path" --> 產生放置每一分類音頻來源檔案(WAV)總目錄之完整路徑位置\n
        '''

        if not os.path.exists(self.Config.Audio_Data_Path):

            raise Exception("The audio data path not found!!")

        else:

            print("[Load_Data]Start load audio data....")

            # print("Audio class quantity: {}".format(
            #     len(os.listdir(self.Config.Audio_Data_Path))))

            test_first = True

            ''' 判斷放置紀錄檔案目錄位置內是否存在有目錄 '''
            if os.listdir(os.path.join(os.getcwd(), self.Config.Log_DirectoryName, self.Config.Log_FeatureData_DirectoryName)) != 0:

                ''' 讀取放置紀錄檔案目錄位置內容 '''
                for read_item in os.listdir(os.path.join(os.getcwd(), self.Config.Log_DirectoryName, self.Config.Log_FeatureData_DirectoryName)):

                    ''' 判斷當前讀取放置紀錄檔案目錄位置內容項目是否為目錄 '''
                    if os.path.isdir(os.path.join(os.getcwd(), self.Config.Log_DirectoryName, self.Config.Log_FeatureData_DirectoryName, read_item)):

                        ''' 讀取當前放置紀錄檔案目錄位置內容目錄內檔案 '''
                        for read_file in os.listdir(os.path.join(os.getcwd(), self.Config.Log_DirectoryName, self.Config.Log_FeatureData_DirectoryName, read_item)):

                            # 刪除當前讀取到檔案
                            os.remove(
                                os.path.join(
                                    os.getcwd(),
                                    self.Config.Log_DirectoryName,
                                    self.Config.Log_FeatureData_DirectoryName,
                                    # 當前讀取放置紀錄檔案目錄位置內容目錄
                                    read_item,
                                    # 當前讀取放置紀錄檔案目錄位置內容目錄內檔案
                                    read_file
                                )
                            )

                        # 刪除當前目錄
                        os.rmdir(
                            os.path.join(
                                os.getcwd(),
                                self.Config.Log_DirectoryName,
                                self.Config.Log_FeatureData_DirectoryName,
                                # 當前讀取放置紀錄檔案目錄位置內容目錄
                                read_item
                            )
                        )

                    ''' 判斷當前讀取放置紀錄檔案目錄位置內容項目是否為檔案 '''
                    if os.path.isfile(os.path.join(os.getcwd(), self.Config.Log_DirectoryName, self.Config.Log_FeatureData_DirectoryName, read_item)):

                        # 刪除當前讀取到檔案
                        os.remove(
                            os.path.join(
                                os.getcwd(),
                                self.Config.Log_DirectoryName,
                                self.Config.Log_FeatureData_DirectoryName,
                                # 當前讀取放置紀錄檔案目錄位置檔案
                                read_item,
                            )
                        )

            ''' 讀取當前指定位置內容 '''
            for read_dir in os.listdir(self.Config.Audio_Data_Path):

                ''' 判斷當前的路徑是否為一個目錄 '''
                if os.path.isdir(os.path.join(self.Config.Audio_Data_Path, read_dir)):

                    # print("[Load_Data]{}".format(read_dir))
                    # print("[Load_Data]{}".format(
                    #     os.listdir(os.path.join(
                    #         self.Config.Audio_Data_Path, read_dir))
                    # ))

                    print(
                        "[Load_Data]Current reading class name：['{}'].".format(
                            read_dir
                        )
                    )

                    ''' 判斷當前讀取目錄內檔案數量是否大於指定數量 '''
                    if len(os.listdir(os.path.join(self.Config.Audio_Data_Path, read_dir))) > self.Config.data_quantity_max:

                        # 增加類別大項數量
                        self.Config.class_num += 1

                        # 串接當前目錄名稱作為類別名稱大項
                        self.class_labels.append(
                            str(read_dir).rstrip(" ").lower())

                        ''' 判斷放置紀錄檔案目錄位置是否不存在 '''
                        if not os.path.exists(os.path.join(os.getcwd(), self.Config.Log_DirectoryName, self.Config.Log_FeatureData_DirectoryName, read_dir)):

                            # 建立放置紀錄檔案目錄
                            os.mkdir(
                                os.path.join(
                                    os.getcwd(),
                                    self.Config.Log_DirectoryName,
                                    self.Config.Log_FeatureData_DirectoryName,
                                    read_dir  # 類別名稱
                                )
                            )

                        ''' 讀取指定數量之下每一個檔案 '''
                        for r_index in range(0, self.Config.data_quantity_max):

                            # 擷取語音wav檔案MFCC特徵值
                            mfcc = self.gen_wav_mfcc(
                                os.path.join(
                                    # 檔案處在目錄位置
                                    os.path.join(
                                        self.Config.Audio_Data_Path,
                                        read_dir,
                                    ),
                                    # 檔案名稱
                                    os.listdir(os.path.join(
                                        self.Config.Audio_Data_Path,
                                        read_dir,
                                    ))[r_index]
                                ),
                                self.Config.sample_rate,
                                self.Config.max_pad_len
                            )

                            # 串接當前語音檔案MFCC特徵值
                            # self.Config.MFCC_Data.append(mfcc)

                            # 串接當前語音類別名稱
                            # self.Config.labels.append(
                            #     read_dir.rstrip(" ").lower())

                            # 輸出語音特徵MFCC紀錄
                            self.__ouputlog_audio_feature(
                                os.path.join(
                                    os.getcwd(),
                                    self.Config.Log_DirectoryName,
                                    self.Config.Log_FeatureData_DirectoryName,
                                    # 類別名稱
                                    read_dir,
                                    # 檔案名稱
                                    str(str(os.listdir(os.path.join(
                                        self.Config.Audio_Data_Path,
                                        read_dir,
                                    ))[r_index]).split(".")[0] + "." + self.Config.log_file_type)
                                ),
                                mfcc
                            )

                            # if test_first:

                            #     test_first = False

                            # mfcc = self.gen_wav_mfcc(
                            #     os.path.join(
                            #         # 檔案處在目錄位置
                            #         os.path.join(
                            #             self.Config.Audio_Data_Path,
                            #             read_dir,
                            #         ),
                            #         # 檔案名稱
                            #         os.listdir(os.path.join(
                            #             self.Config.Audio_Data_Path,
                            #             read_dir,
                            #         ))[r_index]
                            #     ),
                            #     16000,
                            #     11
                            # )

                    else:

                        print("[Load_Data]The ['{}'] data quantity: {} < {}(limit max quantity)".format(
                            read_dir,
                            len(os.listdir(os.path.join(
                                self.Config.Audio_Data_Path, read_dir
                            ))),
                            self.Config.data_quantity_max
                        ))

            class_labels_length = len(self.class_labels)

            print("\n[Load_Data]Audio class quantity: {}".format(
                class_labels_length))

            # print("[Load_Data]\n{}".format(
            #     self.class_labels), end="\n\n")

            ''' 輸出顯示所有的類別與其對應編號 '''
            self.__show_class_num_info(
                # 給予類別編碼後結果
                enumerate(self.class_labels),
                class_labels_length
            )

            print("[Load_Data]Save ['{}'] to [{}]".format(
                self.Config.Log_ClassLabelsData_name,
                os.path.join(
                    os.getcwd(),
                    self.Config.Log_DirectoryName,
                    self.Config.Log_ClassLabelsData_DirectoryName,
                    # 檔案名稱
                    self.Config.Log_ClassLabelsData_name
                )
            ))

            ''' 判斷放置總類別與對應編號內容檔案是否存在 '''
            if os.path.exists(os.path.join(os.getcwd(), self.Config.Log_DirectoryName, self.Config.Log_ClassLabelsData_DirectoryName, self.Config.Log_ClassLabelsData_name)):

                # 移除指定檔案
                os.remove(
                    os.path.join(
                        os.getcwd(),
                        self.Config.Log_DirectoryName,
                        self.Config.Log_ClassLabelsData_DirectoryName,
                        # 檔案名稱
                        self.Config.Log_ClassLabelsData_name
                    )
                )

            # 輸出總類別與對應編號內容
            self.__ouputlog_audio_class(
                os.path.join(
                    os.getcwd(),
                    self.Config.Log_DirectoryName,
                    self.Config.Log_ClassLabelsData_DirectoryName,
                    # 檔案名稱
                    self.Config.Log_ClassLabelsData_name
                ),

                # 給予類別編碼後結果
                enumerate(self.class_labels)
            )
