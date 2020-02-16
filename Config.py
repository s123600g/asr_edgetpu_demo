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

import os
import numpy as np


class Config(object):
    '''
    ASR project basic argument settings
    '''

    ''' 
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    data sources settings
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    # Data sources for wav file settings  ,  directory name and  path
    Audio_Data_Directory_Root = "/home/jyu/Data"
    Audio_Data_DirectoryName = "audio_data"
    # Generate a full data sources path
    Audio_Data_Path = os.path.join(
        Audio_Data_Directory_Root,
        Audio_Data_DirectoryName
    )

    # Data sources for prediction file settings , directory name and  path
    Prediction_Audio_Data_Directory_Root = "/home/jyu/Data"
    Prediction_Audio_Data_DirectoryName = "prediction_data"
    # Generate a full data sources path
    Prediction_Audio_Data_Path = os.path.join(
        Prediction_Audio_Data_Directory_Root,
        Prediction_Audio_Data_DirectoryName
    )
    '''
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''

    ''' 
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    wav data basic argument  settings
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    data_quantity_max = 450  # limit wav data quantity for  each class
    sample_rate = 16000
    max_pad_len = 11
    class_num = 0  # class number start value
    channel = 1  # single channel
    labels = []
    MFCC_Data = []
    Train_DataSet = []
    Train_Labels = []
    Test_DataSet = []
    Test_Labels = []
    Valid_DataSet = []
    Valid_Labels = []
    data_row = 0
    data_column = 0
    '''
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''

    ''' 
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    SQLite DB  settings
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    SQLite_DB_DirectoryName = "DB"
    SQLite_name = "class.db3"
    db_TableName = 'audioclass'
    column_ClassNum = 'ClassNum'
    column_Classname = 'ClassName'
    '''
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''

    ''' 
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    ouput log  settings
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    # general ouput log
    Log_DirectoryName = "log_file"
    log_file_type = "txt"
    Log_FeatureData_DirectoryName = "audio_feature"
    Log_ClassLabelsData_DirectoryName = "audio_classlabels"
    Log_Recognition_Result_DirectoryName = "recognition_result"
    Log_ClassLabelsData_name = str(
        Log_ClassLabelsData_DirectoryName + "." + log_file_type
    )

    # for TensorBoard
    Log_TensorBoard_DirectoryName = "TensorBoard"
    Log_TensorBoard_Path = os.path.join(
        os.getcwd(), Log_DirectoryName, Log_TensorBoard_DirectoryName, "logs"
    )

    # 輸出訓練過程變化圖檔案放置目錄
    Plot_Figure_DirectoryName = "plot_figure"
    '''
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''

    ''' 
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    model  settings
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    # 訓練後模型儲存參數設置
    Model_DirectoryName = "model"
    Model_Name = "Speech_Recognition_Model.h5"
    Model_Weight_Name = "Speech_Recognition_Weight.h5"
    Model_Path = os.path.join(
        os.getcwd(),
        Model_DirectoryName,
        Model_Name
    )
    Model_Weight_Path = os.path.join(
        os.getcwd(),
        Model_DirectoryName,
        Model_Weight_Name
    )
    Model_checkpoints_DirectoryName = "checkpoints"
    Model_checkpoints_Path = os.path.join(
        os.getcwd(),
        Model_checkpoints_DirectoryName,
        Model_checkpoints_DirectoryName
    )

    # 轉換模型參數設置
    Input_Model_Path = os.path.join(os.getcwd(), "model")
    Output_Model_Name = "ASR_Model.tflite"
    Output_Model_Path = os.path.join(
        os.getcwd(), "tflite_model",
        Output_Model_Name
    )

    #  model checkpoint
    Model_ModelCheckpoint_DirectoryName = "ModelCheckpoint"
    Model_ModelCheckpoint_Path = os.path.join(
        os.getcwd(),
        Model_DirectoryName,
        Model_ModelCheckpoint_DirectoryName
    )

    # model save path
    Model_PB_DirectoryName = "model_pb"
    Model_PB_Name = "frozen_model.pb"
    Model_PB_Path = os.path.join(
        os.getcwd(),
        Model_DirectoryName,
        Model_PB_DirectoryName,
        Model_PB_Name
    )

    # model layers name , input layer and ouput layer
    input_arrays = ["conv2d_input"]
    output_arrays = ["dense_1/Softmax"]
    '''
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
