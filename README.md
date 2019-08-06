[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

# 使用EdgeTpu應用在語音模型預測之簡單實例

使用資料集資訊
--
使用在Tensorflow官方 Simple Audio Recognition 例子內所使用之語音資料集speech_commands_v0.02.tar.gz<br/>
[Simple Audio Recognition](https://www.tensorflow.org/tutorials/sequences/audio_recognition)<br/>
[speech_commands_v0.02.tar.gz](https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz)<br/>
<p></p>

#### 本專案所使用資料集，為35個類別，每一個類別各有450個語音特徵，總共有15750個語音特徵資料，連結如下：<br/>
1. 已事先擷取出語音特徵並存成文字檔：
[audio_feature_txt](https://drive.google.com/open?id=11X-vlDNjCH4t98fRs5reBuSXolAUeh7b)<br/>
如果要使用此資料集，請在專案目錄內log_file/建立一個audio_feature目錄，將壓縮檔內audio_feature/底下所有檔案目錄移動到audio_feature底下。

2.  原始語音wav檔：
[audio_feature_wav](https://drive.google.com/open?id=1xnBpX8WsJtV2hbcY90O0Pw17sXZmpi80)<br/>
如果要使用此資料集，請在專案目錄內建立一個audio_data目錄，將壓縮檔內audio_data/底下所有檔案目錄移動到audio_data底下。

 #### 關於語音特徵值：<br>
 使用 [Python - Librosa](https://librosa.github.io/librosa/install.html) <br/>
 讀取音檔內容使用 --> librosa.core.load <br/>
 擷取MFCC特徵 --> librosa.feature.mfcc <br/>

slides
--
[使用EdgeTpu應用在語音模型預測之簡單實例](https://docs.google.com/presentation/d/1ymZKADuUzkwwKdbGHvZ2SpFxg3LBCAEO4x32p3buI3U/edit?usp=sharing)

Medium 文章
--
1. [使用EdgeTpu應用在語音模型預測之簡單實例(一)-前言與開發環境配置](https://medium.com/@s123600g/%E4%BD%BF%E7%94%A8edgetpu%E6%87%89%E7%94%A8%E5%9C%A8%E8%AA%9E%E9%9F%B3%E6%A8%A1%E5%9E%8B%E9%A0%90%E6%B8%AC%E4%B9%8B%E7%B0%A1%E5%96%AE%E5%AF%A6%E4%BE%8B-%E4%B8%80-%E5%89%8D%E8%A8%80%E8%88%87%E9%96%8B%E7%99%BC%E7%92%B0%E5%A2%83%E9%85%8D%E7%BD%AE-d8720eb0d970)<br/>
2. [使用EdgeTpu應用在語音模型預測之簡單實例(二)-語音資料集處理](https://medium.com/@s123600g/%E4%BD%BF%E7%94%A8edgetpu%E6%87%89%E7%94%A8%E5%9C%A8%E8%AA%9E%E9%9F%B3%E6%A8%A1%E5%9E%8B%E9%A0%90%E6%B8%AC%E4%B9%8B%E7%B0%A1%E5%96%AE%E5%AF%A6%E4%BE%8B-%E4%BA%8C-%E8%AA%9E%E9%9F%B3%E8%B3%87%E6%96%99%E9%9B%86%E8%99%95%E7%90%86-a9a1f4492bc0)<br/>
3. [使用EdgeTpu應用在語音模型預測之簡單實例(三)-建立模型與訓練](https://medium.com/@s123600g/%E4%BD%BF%E7%94%A8edgetpu%E6%87%89%E7%94%A8%E5%9C%A8%E8%AA%9E%E9%9F%B3%E6%A8%A1%E5%9E%8B%E9%A0%90%E6%B8%AC%E4%B9%8B%E7%B0%A1%E5%96%AE%E5%AF%A6%E4%BE%8B-%E4%B8%89-%E5%BB%BA%E7%AB%8B%E6%A8%A1%E5%9E%8B%E8%88%87%E8%A8%93%E7%B7%B4-3ae20b170eb)<br/>
4. [使用EdgeTpu應用在語音模型預測之簡單實例(四)-模型轉換格式為tflite](https://medium.com/@s123600g/%E4%BD%BF%E7%94%A8edgetpu%E6%87%89%E7%94%A8%E5%9C%A8%E8%AA%9E%E9%9F%B3%E6%A8%A1%E5%9E%8B%E9%A0%90%E6%B8%AC%E4%B9%8B%E7%B0%A1%E5%96%AE%E5%AF%A6%E4%BE%8B-%E5%9B%9B-%E6%A8%A1%E5%9E%8B%E8%BD%89%E6%8F%9B%E6%A0%BC%E5%BC%8F%E7%82%BAtflite-3cd1b3c2b122)<br/>
5. [使用EdgeTpu應用在語音模型預測之簡單實例(五)-使用 edgetpu_compiler 轉換 EdgeTpu 可識別 tflite模型](https://medium.com/@s123600g/%E4%BD%BF%E7%94%A8edgetpu%E6%87%89%E7%94%A8%E5%9C%A8%E8%AA%9E%E9%9F%B3%E6%A8%A1%E5%9E%8B%E9%A0%90%E6%B8%AC%E4%B9%8B%E7%B0%A1%E5%96%AE%E5%AF%A6%E4%BE%8B-%E4%BA%94-%E4%BD%BF%E7%94%A8-edgetpu-compiler-%E8%BD%89%E6%8F%9B-edgetpu-%E5%8F%AF%E8%AD%98%E5%88%A5-tflite%E6%A8%A1%E5%9E%8B-54fdf75e25a3)<br/>
6. [使用EdgeTpu應用在語音模型預測之簡單實例(六)-進行 EdgeTpu 模型預測](https://medium.com/@s123600g/%E4%BD%BF%E7%94%A8edgetpu%E6%87%89%E7%94%A8%E5%9C%A8%E8%AA%9E%E9%9F%B3%E6%A8%A1%E5%9E%8B%E9%A0%90%E6%B8%AC%E4%B9%8B%E7%B0%A1%E5%96%AE%E5%AF%A6%E4%BE%8B-%E5%85%AD-%E9%80%B2%E8%A1%8C-edgetpu-%E6%A8%A1%E5%9E%8B%E9%A0%90%E6%B8%AC-e76bf901eecc)<br/>
