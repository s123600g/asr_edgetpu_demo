[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

# 使用EdgeTpu應用在語音模型預測之簡單實例

使用資料集資訊
--
使用在Tensorflow官方 Simple Audio Recognition 例子內所使用之語音資料集speech_commands_v0.02.tar.gz<br/>
[Simple Audio Recognition](https://www.tensorflow.org/tutorials/sequences/audio_recognition)<br/>
[speech_commands_v0.02.tar.gz](https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz)<br/>
<p></p>

#### 本專案所使用資料集，為35個類別，每一個類別各有450個語音特徵，總共有15750個語音特徵資料，連結如下：<br/>
1. 已事先擷取出語音特徵並存成文字檔
[audio_feature_txt](https://drive.google.com/open?id=11X-vlDNjCH4t98fRs5reBuSXolAUeh7b)<br/>
如果要使用此資料集，請在專案目錄內log_file/建立一個audio_feature目錄，將壓縮檔內audio_feature/底下所有檔案移動到audio_feature目錄。

2.  原始語音wav檔
[audio_feature_wav](https://drive.google.com/open?id=1xnBpX8WsJtV2hbcY90O0Pw17sXZmpi80)<br/>
如果要使用此資料集，請在專案目錄內建立一個audio_data目錄，將壓縮檔內audio_data/底下所有檔案移動到audio_data目錄。


Medium 文章
--
1. [使用EdgeTpu應用在語音模型預測之簡單實例(一)-前言與開發環境配置]()<br/>
2. [使用EdgeTpu應用在語音模型預測之簡單實例(二)-語音資料集處理]()<br/>
3. [使用EdgeTpu應用在語音模型預測之簡單實例(三)-建立模型與訓練]()<br/>
4. [使用EdgeTpu應用在語音模型預測之簡單實例(四)-模型轉換格式為tflite]()<br/>
5. [使用EdgeTpu應用在語音模型預測之簡單實例(五)-使用 edgetpu_compiler 轉換 EdgeTpu 可識別 tflite模型]()<br/>
6. [使用EdgeTpu應用在語音模型預測之簡單實例(六)-進行 EdgeTpu 模型預測]()<br/>
