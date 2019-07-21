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

from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras import backend as K

import keras
import os
import numpy as np
import random
import sqlite3


class Gen_DataSet:

    def __init__(self, Config):

        self.Config = Config

        self.class_label = dict()

        self.dbconn = sqlite3.connect(
            os.path.join(
                os.getcwd(),
                self.Config.SQLite_DB_DirectoryName,
                self.Config.SQLite_name
            )
        )

        self.curs = self.dbconn.cursor()

    def __SQL_Insert_Class_Labels(self):
        '''
        新增分類標籤與對應編號進SQLite DB。\n
        關於SQLite DB參數設置，可在"Config.py"內找到SQLite DB 參數配置區塊，有關參數資訊如下： \n

        "SQLite_DB_DirectoryName" --> 專案資料夾內放置SQLite DB單檔位置 \n
        "SQLite_name" --> SQLite DB單檔名稱 \n 
        "db_TableName" --> 使用之資料庫內資料表 \n
        "column_ClassNum" --> 使用之資料庫內資料表內資料欄位名稱，這裡是使用"ClassNum"欄位名稱 \n
        "column_Classname" --> 使用之資料庫內資料表內資料欄位名稱，這裡是使用"ClassName"欄位名稱 \n
        '''

        print("[Gen_DataSet]Start scanning class....", end="\n\n")

        ''' Run SQL DB Table Clearn '''
        SQL_delete_syntax = '''
        DELETE FROM {}
        '''.format(self.Config.db_TableName)
        SQL_run = self.curs.execute(SQL_delete_syntax)

        if SQL_run:

            self.dbconn.commit()

        else:

            raise Exception("[Gen_DataSet]Run SQL_Delete_Syntax Faild !!")

        temp_read_content = ""

        ''' 判斷放置已編號之類別內容文字檔是否存在 '''
        if os.path.exists(os.path.join(os.getcwd(), self.Config.Log_DirectoryName, self.Config.Log_ClassLabelsData_DirectoryName, self.Config.Log_ClassLabelsData_name)):

            ''' 讀取已編號之類別內容文字檔 '''
            with open(os.path.join(os.getcwd(), self.Config.Log_DirectoryName, self.Config.Log_ClassLabelsData_DirectoryName, self.Config.Log_ClassLabelsData_name), "r") as read:

                temp_read_content = read.readlines()

            # 讀取擷取出來內容
            for read_data in temp_read_content:

                read_data = str(read_data).rstrip("\n").split(" ")

                ''' Run SQL_Insert_Syntax '''
                SQL_Insert_syntax = '''
                INSERT INTO {}('{}','{}')
                VALUES('{}','{}')
                '''.format(
                    self.Config.db_TableName,
                    self.Config.column_ClassNum,
                    self.Config.column_Classname,
                    int(read_data[0]),
                    str(read_data[1]))
                SQL_run = self.curs.execute(SQL_Insert_syntax)

                if SQL_run:

                    self.dbconn.commit()
                    self.class_label[int(read_data[0])] = str(read_data[1])
                    self.Config.class_num += 1

                else:

                    raise Exception(
                        "[Gen_DataSet]Run SQL_Insert_Syntax Faild !!")

            print("[Gen_DataSet]Class label：{}".format(
                self.class_label
            ))

        else:

            raise Exception("[Gen_DataSet] The file ['{}'] can not found !!".format(
                self.Config.Log_ClassLabelsData_name))

    def __load_DataSet_File(self):
        '''
        讀取指定的目錄內所有分類內特徵值檔案，此方法最後會產生所有特徵值內容矩陣與分類標籤矩陣。\n

        讀取指定的目錄設置，可在Config.py內針對輸出檔案參數配置配置區塊，進行讀取資料來源配置位置。\n
        特徵值檔案與分類標籤資訊統一放在"Log_DirectoryName"指定的目錄名稱內，關於內部目錄結構參數設置如下：\n
        "Log_FeatureData_DirectoryName" --> 放置每一分類特徵值檔案總目錄 \n
        "Log_ClassLabelsData_DirectoryName" --> 放置分類標籤資訊與對應編號檔案總目錄 \n

        '''

        print("\n[Gen_DataSet]Start Reading DataSet File....", end="\n\n")

        temp_read_content = ""
        temp_mfcc_feature = list()

        ''' 讀取放置所有分類語音特徵檔目錄內容 '''
        for read_classfloder in os.listdir(os.path.join(os.getcwd(), self.Config.Log_DirectoryName, self.Config.Log_FeatureData_DirectoryName)):

            print(
                "[Gen_DataSet]Current reading class name：['{}'].".format(
                    read_classfloder
                )
            )

            ''' 讀取放置所有分類語音特徵檔目錄內容-每一個分類目錄內檔案 '''
            for read_file in os.listdir(os.path.join(os.getcwd(), self.Config.Log_DirectoryName, self.Config.Log_FeatureData_DirectoryName, read_classfloder)):

                ''' 判斷每一個分類目錄內放置語音特徵文字檔目錄是否存在 '''
                if os.path.exists(os.path.join(os.getcwd(), self.Config.Log_DirectoryName, self.Config.Log_FeatureData_DirectoryName, read_classfloder, read_file)):

                    # print("{}".format(read_file))

                    # 串接每一個音頻特徵檔案對應分類位置
                    self.Config.labels.append(read_classfloder)

                    ''' 讀取放置語音特徵檔案內容 '''
                    with open(os.path.join(os.getcwd(), self.Config.Log_DirectoryName, self.Config.Log_FeatureData_DirectoryName, read_classfloder, read_file), "r") as read:

                        temp_read_content = read.readlines()

                    ''' 將讀取內容作處理儲存至特徵清單 '''
                    for read_row in temp_read_content:

                        # print("{}".format(read_item))

                        # 將每一行先做去除尾巴換行符號，並進行資料切割
                        read_row = str(read_row).rstrip(" \n").split(" ")

                        temp_list = list()

                        # 讀取已完成去除尾巴換行符號與進行資料切割之每列內之欄位資料
                        for feature in read_row:

                            # 存放每一列的欄位資料
                            temp_list.append(float(feature))

                        # 存放一整列資料
                        temp_mfcc_feature.append(temp_list)

                    # 串接存放每一個音頻特徵檔案內容
                    self.Config.MFCC_Data.append(temp_mfcc_feature)

                    temp_mfcc_feature = list()

        # print("[Gen_DataSet]Audio class quantity: {}".format(
        #     len(
        #         os.listdir(
        #             os.path.join(
        #                 os.getcwd(),
        #                 self.Config.Log_DirectoryName,
        #                 self.Config.Log_FeatureData_DirectoryName
        #             )
        #         )
        #     )
        # ))

        # 將存放所有分類音頻特徵內容清單轉為numpy array型態
        self.Config.MFCC_Data = np.array(self.Config.MFCC_Data)

        # 將存放所有分類音頻對應分類位置清單轉為numpy array型態
        self.Config.labels = np.array(self.Config.labels)

        print("[Gen_DataSet] MFCC_Data type: {}".format(
            type(self.Config.MFCC_Data)))

        print("[Gen_DataSet] MFCC_Data shape: {}".format(
            self.Config.MFCC_Data.shape))

        print("[Gen_DataSet] labels_data type: {}".format(
            type(self.Config.labels)))

        print("[Gen_DataSet] labels_data shape: {}".format(
            self.Config.labels.shape))

        print("[Gen_DataSet] Class quantity: {}".format(
            self.Config.class_num))

        print()

        # print("{}".format(
        #     self.Config.MFCC_Data[0]
        # ))

    def __shuffle_DataSet(self):
        '''
        將全部所有讀取的特徵值檔案內容與所屬分類標籤，進行資料矩陣內部打散排序。
        '''

        # shuffle data
        perm_array = np.arange(len(self.Config.MFCC_Data))
        np.random.shuffle(perm_array)
        self.Config.MFCC_Data = self.Config.MFCC_Data[perm_array]
        self.Config.labels = self.Config.labels[perm_array]

        # print("[Gen_DataSet]Labels：\n{}".format(
        #     self.Config.labels
        # ))

    def __char_to_code(self):
        '''
        將所讀取的所有特徵對應分類標籤資訊矩陣內部，進行轉換成已分好之分類標籤對應編號。 \n

        例如：\n

        假設所有分類標籤對應編號資訊如下：\n
        {0: 'backward', 1: 'bed', 2: 'bird', 3: 'cat', 4: 'dog', 5: 'down', 6: 'eight', 7: 'five', 8: 'follow', 9: 'forward', 10: 'four', 11: 'go', 12: 'happy', 13: 'house', 14: 'learn', 15: 'left', 16: 'marvin', 17: 'nine', 18: 'no', 19: 'off', 20: 'on', 21: 'one', 22: 'right', 23: 'seven', 24: 'sheila', 25: 'six', 26: 'stop', 27: 'three', 28: 'tree', 29: 'two', 30: 'up', 31: 'visual', 32: 'wow', 33: 'yes', 34: 'zero'}\n\n

        原始所有特徵對應分類標籤資訊矩陣內容如下： \n
        ['wow', 'nine', 'five', ... ,'dog' ,'tree', 'forward'] \n \n
        轉換分類標籤對應編號後之所有特徵對應分類標籤資訊矩陣內容如下：\n
        [32 ,17,7, ... ,4,28,9]\n
        '''

        for index in list(self.class_label.keys()):

            # print("{} {}".format(index, self.class_label[index]))

            self.Config.labels = np.array(
                [int(index) if label == self.class_label[index]
                 else label for label in self.Config.labels]
            )

        # print("[Gen_DataSet]Labels：\n{}".format(
        #     self.Config.labels
        # ))

    def __split_Train_Test_Valid_Data(self):
        '''
        進行資料集產生，分別有訓練、驗證、測試資料集，各自分別有放置特徵數據(DataSet)與分類標籤(Labels)。\n

        處理步驟如下：\n
        Step1.切割資料集，產生訓練與驗證資料集\n
        Step2.切割資料集，產生測試資料集 \n
        Step3.將訓練、驗證、測試資料集轉換成numpy array型態 \n
        Step4.將訓練、驗證、測試資料集小數精度型態轉換 \n
        Step5.將訓練資料集向量維度轉換(Reshape) \n
        '''

        # 切割資料集，產生訓練與驗證資料集
        self.Config.Train_DataSet, self.Config.Valid_DataSet, self.Config.Train_Labels, self.Config.Valid_Labels = train_test_split(
            self.Config.MFCC_Data,  # 訓練特徵資料集來源
            self.Config.labels,  # 訓練分類標籤資料集來源
            test_size=0.2,  # 資料集分割比例
            random_state=random.randint(0, self.Config.class_num)
        )

        # 切割資料集，產生測試資料集
        _, self.Config.Test_DataSet, _, self.Config.Test_Labels = train_test_split(
            self.Config.Train_DataSet,  # 訓練特徵資料集來源
            self.Config.Train_Labels,  # 訓練分類標籤資料集來源
            test_size=0.2,  # 資料集分割比例
            random_state=random.randint(0, len(self.Config.Train_DataSet))
        )

        # 將訓練、驗證、測試資料集轉換成numpy array型態
        self.Config.Train_DataSet = np.array(self.Config.Train_DataSet)
        self.Config.Train_Labels = np.array(self.Config.Train_Labels)
        self.Config.Valid_DataSet = np.array(self.Config.Valid_DataSet)
        self.Config.Valid_Labels = np.array(self.Config.Valid_Labels)
        self.Config.Test_DataSet = np.array(self.Config.Test_DataSet)
        self.Config.Test_Labels = np.array(self.Config.Test_Labels)

        # 將訓練、驗證、測試資料集小數維度轉換
        self.Config.Train_DataSet = self.Config.Train_DataSet.astype('float32')
        self.Config.Valid_DataSet = self.Config.Valid_DataSet.astype('float32')
        self.Config.Test_DataSet = self.Config.Test_DataSet.astype('float32')

        ''' 將訓練資料集維度轉換 '''
        self.Config.Train_DataSet = self.Config.Train_DataSet.reshape(
            self.Config.Train_DataSet.shape[0],  # 資料總筆數
            self.Config.Train_DataSet.shape[1],  # 資料矩陣之列
            self.Config.Train_DataSet.shape[2],  # 資料矩陣之欄
            self.Config.channel   # 單一通道(single channel)
        )

        ''' 將驗證資料集維度轉換 '''
        self.Config.Valid_DataSet = self.Config.Valid_DataSet.reshape(
            self.Config.Valid_DataSet.shape[0],  # 資料總筆數
            self.Config.Valid_DataSet.shape[1],  # 資料矩陣之列
            self.Config.Valid_DataSet.shape[2],  # 資料矩陣之欄
            self.Config.channel  # 單一通道(single channel)
        )

        ''' 將測試資料集維度轉換 '''
        self.Config.Test_DataSet = self.Config.Test_DataSet.reshape(
            self.Config.Test_DataSet.shape[0],  # 資料總筆數
            self.Config.Test_DataSet.shape[1],  # 資料矩陣之列
            self.Config.Test_DataSet.shape[2],  # 資料矩陣之欄
            self.Config.channel   # 單一通道(single channel)
        )

        print("[Gen_DataSet] Train_DataSet shape：{}".format(
            self.Config.Train_DataSet.shape)
        )
        print("[Gen_DataSet] Valid_DataSet shape：{}".format(
            self.Config.Valid_DataSet.shape)
        )
        print("[Gen_DataSet] Test_DataSet shape：{}".format(
            self.Config.Test_DataSet.shape)
        )

    def __one_hot_process(self):
        '''
        將資料集(訓練、驗證、測試)分類標籤編號，進行One-Hot編碼。
        '''

        self.Config.Train_Labels = np_utils.to_categorical(
            self.Config.Train_Labels,
            self.Config.class_num
        )

        self.Config.Valid_Labels = np_utils.to_categorical(
            self.Config.Valid_Labels,
            self.Config.class_num
        )

        self.Config.Test_Labels = np_utils.to_categorical(
            self.Config.Test_Labels,
            self.Config.class_num
        )

        # print("[Gen_DataSet]Train Labels：\n {}".format(
        #     self.Config.Train_Labels
        # ))

        # print("[Gen_DataSet]Valid Labels：\n {}".format(
        #     self.Config.Valid_Labels
        # ))

        # print("[Gen_DataSet]Test Labels：\n {}".format(
        #     self.Config.Test_Labels
        # ))

    def DataSet_Process(self):
        '''
        訓練模型所需資料集處理產生總方法。

        處理程序步驟如下：\n
        Step1. 執行資料分類標籤之對應編號存進資料庫\n
        Step2. 讀取語音特徵值檔案，取得每一筆特徵值內容與分類標籤\n
        Step3. 將所有已讀取完每一筆特徵值內容與分類標籤進行資料混淆矩陣打散\n
        Step4. 將每一筆特徵值所屬分類標籤進行轉換成對應編號，與資料庫每一分類所屬編號內容相對應\n
        Step5. 產生訓練模型所需資料集(訓練、驗證、測試)\n
        Step6. 將所有資料集(訓練、驗證、測試)之分類標籤編號進行One-Hot 編碼\n

        '''

        ''' 執行資料分類標籤之對應編號存進資料庫 '''
        self.__SQL_Insert_Class_Labels()

        ''' 讀取語音特徵值檔案，取得每一筆特徵值內容與分類標籤 '''
        self.__load_DataSet_File()

        ''' 將所有已讀取完每一筆特徵值內容與分類標籤進行資料混淆矩陣打散 '''
        self.__shuffle_DataSet()

        ''' 將每一筆特徵值所屬分類標籤進行轉換成對應編號，與資料庫每一分類所屬編號內容相對應 '''
        self.__char_to_code()

        ''' 產生訓練模型所需資料集(訓練、驗證、測試) '''
        self.__split_Train_Test_Valid_Data()

        ''' 將所有資料集(訓練、驗證、測試)之分類標籤編號進行One-Hot 編碼 '''
        self.__one_hot_process()
