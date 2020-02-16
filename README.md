[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

參考資訊
--
1. slides<br/>
[使用EdgeTpu應用在語音模型預測之簡單實例](https://docs.google.com/presentation/d/1ymZKADuUzkwwKdbGHvZ2SpFxg3LBCAEO4x32p3buI3U/edit?usp=sharing)

2. Medium 文章<br/>
[1] [使用EdgeTpu應用在語音模型預測之簡單實例(一)-前言與開發環境配置](https://medium.com/@s123600g/%E4%BD%BF%E7%94%A8edgetpu%E6%87%89%E7%94%A8%E5%9C%A8%E8%AA%9E%E9%9F%B3%E6%A8%A1%E5%9E%8B%E9%A0%90%E6%B8%AC%E4%B9%8B%E7%B0%A1%E5%96%AE%E5%AF%A6%E4%BE%8B-%E4%B8%80-%E5%89%8D%E8%A8%80%E8%88%87%E9%96%8B%E7%99%BC%E7%92%B0%E5%A2%83%E9%85%8D%E7%BD%AE-d8720eb0d970)<br/>
[2] [使用EdgeTpu應用在語音模型預測之簡單實例(二)-語音資料集處理](https://medium.com/@s123600g/%E4%BD%BF%E7%94%A8edgetpu%E6%87%89%E7%94%A8%E5%9C%A8%E8%AA%9E%E9%9F%B3%E6%A8%A1%E5%9E%8B%E9%A0%90%E6%B8%AC%E4%B9%8B%E7%B0%A1%E5%96%AE%E5%AF%A6%E4%BE%8B-%E4%BA%8C-%E8%AA%9E%E9%9F%B3%E8%B3%87%E6%96%99%E9%9B%86%E8%99%95%E7%90%86-a9a1f4492bc0)<br/>
[3] [使用EdgeTpu應用在語音模型預測之簡單實例(三)-建立模型與訓練](https://medium.com/@s123600g/%E4%BD%BF%E7%94%A8edgetpu%E6%87%89%E7%94%A8%E5%9C%A8%E8%AA%9E%E9%9F%B3%E6%A8%A1%E5%9E%8B%E9%A0%90%E6%B8%AC%E4%B9%8B%E7%B0%A1%E5%96%AE%E5%AF%A6%E4%BE%8B-%E4%B8%89-%E5%BB%BA%E7%AB%8B%E6%A8%A1%E5%9E%8B%E8%88%87%E8%A8%93%E7%B7%B4-3ae20b170eb)<br/>
[4] [使用EdgeTpu應用在語音模型預測之簡單實例(四)-模型轉換格式為tflite](https://medium.com/@s123600g/%E4%BD%BF%E7%94%A8edgetpu%E6%87%89%E7%94%A8%E5%9C%A8%E8%AA%9E%E9%9F%B3%E6%A8%A1%E5%9E%8B%E9%A0%90%E6%B8%AC%E4%B9%8B%E7%B0%A1%E5%96%AE%E5%AF%A6%E4%BE%8B-%E5%9B%9B-%E6%A8%A1%E5%9E%8B%E8%BD%89%E6%8F%9B%E6%A0%BC%E5%BC%8F%E7%82%BAtflite-3cd1b3c2b122)<br/>
[5] [使用EdgeTpu應用在語音模型預測之簡單實例(五)-使用 edgetpu_compiler 轉換 EdgeTpu 可識別 tflite模型](https://medium.com/@s123600g/%E4%BD%BF%E7%94%A8edgetpu%E6%87%89%E7%94%A8%E5%9C%A8%E8%AA%9E%E9%9F%B3%E6%A8%A1%E5%9E%8B%E9%A0%90%E6%B8%AC%E4%B9%8B%E7%B0%A1%E5%96%AE%E5%AF%A6%E4%BE%8B-%E4%BA%94-%E4%BD%BF%E7%94%A8-edgetpu-compiler-%E8%BD%89%E6%8F%9B-edgetpu-%E5%8F%AF%E8%AD%98%E5%88%A5-tflite%E6%A8%A1%E5%9E%8B-54fdf75e25a3)<br/>
[6] [使用EdgeTpu應用在語音模型預測之簡單實例(六)-進行 EdgeTpu 模型預測](https://medium.com/@s123600g/%E4%BD%BF%E7%94%A8edgetpu%E6%87%89%E7%94%A8%E5%9C%A8%E8%AA%9E%E9%9F%B3%E6%A8%A1%E5%9E%8B%E9%A0%90%E6%B8%AC%E4%B9%8B%E7%B0%A1%E5%96%AE%E5%AF%A6%E4%BE%8B-%E5%85%AD-%E9%80%B2%E8%A1%8C-edgetpu-%E6%A8%A1%E5%9E%8B%E9%A0%90%E6%B8%AC-e76bf901eecc)<br/>
[7] [EdgeTpu操作環境安裝配置筆記-使用Google Coral USB Accelerator for Ubuntu](https://medium.com/@s123600g/%E4%BD%BF%E7%94%A8edgetpu%E6%87%89%E7%94%A8%E5%9C%A8%E8%AA%9E%E9%9F%B3%E6%A8%A1%E5%9E%8B%E9%A0%90%E6%B8%AC%E4%B9%8B%E7%B0%A1%E5%96%AE%E5%AF%A6%E4%BE%8B-%E4%B8%80-%E5%89%8D%E8%A8%80%E8%88%87%E9%96%8B%E7%99%BC%E7%92%B0%E5%A2%83%E9%85%8D%E7%BD%AE-d8720eb0d970)<br/>
[8] [EdgeTpu操作環境安裝配置筆記(二)-使用Google Coral USB Accelerator for Windows 10](https://medium.com/@s123600g/edgetpu%E6%93%8D%E4%BD%9C%E7%92%B0%E5%A2%83%E5%AE%89%E8%A3%9D%E9%85%8D%E7%BD%AE%E7%AD%86%E8%A8%98-%E4%BA%8C-9282f2f62812)<br/>

關於資料集
--
使用在Tensorflow官方[Simple Audio Recognition](https://www.tensorflow.org/tutorials/sequences/audio_recognition) 例子內所使用之語音資料集[speech_commands_v0.02.tar.gz](https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz)<br/>

#### 使用資料集為35個類別，每一個類別各有450個語音特徵，總共有15750個語音特徵資料，連結如下：<br/>
1.  [audio_feature_txt](https://drive.google.com/open?id=11X-vlDNjCH4t98fRs5reBuSXolAUeh7b)<br/>
已事先擷取出語音特徵並存成文字檔。如果要使用此資料集，請在專案目錄內log_file/建立一個audio_feature目錄，將壓縮檔內audio_feature/底下所有檔案目錄移動到audio_feature底下。
2.  [audio_feature_wav](https://drive.google.com/open?id=1xnBpX8WsJtV2hbcY90O0Pw17sXZmpi80)<br/>
原始語音wav檔

#### 關於語音特徵值MFCC：<br/>
 使用 [Python - Librosa](https://librosa.github.io/librosa/install.html) <br/>
 讀取音檔內容使用 --> librosa.core.load <br/>
 擷取MFCC特徵 --> librosa.feature.mfcc <br/>
 完整程式碼在 Load_Data.py <br/>

關於專案操作
--
**Step 1. 取得原始語音wav檔**<br/>
請先下載 [audio_feature_wav](https://drive.google.com/open?id=1xnBpX8WsJtV2hbcY90O0Pw17sXZmpi80)，下載完畢會是一個壓縮檔(audio_data.zip)，解壓縮到自己指定位置。<br/>

**Step 2. 設置語音wav檔來源位置**<br/>
在Config.py內 data sources settings部份，設置來源所在根目錄位置目錄名稱。<br/>
```
Audio_Data_Directory_Root = "/home/jyu/Data" # 所在根目錄位置
Audio_Data_DirectoryName = "audio_data" # 目錄名稱
```
**Step 3. 執行語音特徵擷取**<br/>
執行之前請在專案目錄內log_file/底下，建立一個audio_feature空目錄，完成後在執行Gen_Datafile.py。<br/>

> python Gen_Datafile.py

執行完畢之後，會將擷取出來特徵結果文字檔，放在log_file/底下audio_feature目錄，每一個類別結果都會存放在自己分類別名稱命名之目錄，同時會在log_file/audio_classlabels/產生分類對應表(audio_classlabels.txt)<br/>

audio_classlabels.txt：
```
0 happy
1 off
2 bird
3 tree
4 follow
5 up
6 house
7 learn
8 four
9 on
10 backward
11 three
12 no
13 seven
14 left
15 one
16 sheila
17 visual
18 eight
19 down
20 cat
21 five
22 dog
23 two
24 nine
25 go
26 bed
27 marvin
28 yes
29 right
30 zero
31 stop
32 wow
33 forward
34 six
```

**Step 4. 模型建置、訓練、輸出**<br/>
請執行Gen_Datafile.py。<br/>

>  python Train_Data.py

內部執行程序為：載入特徵資料-->資料處理及產生資料集-->建立模型-->訓練模型-->輸出模型-->存放pb模型在model/model_pb/<br/>

**Step 5. pb模型轉換成tflite模型、tflite模型再編譯成edgetpu可識別模型**<br/>
請執行Model_pb_to_tflite.py。<br/>

>  python Model_pb_to_tflite.py

轉換完成tflite模型存放在tflite_model/ASR_Model.tflite
![image](https://github.com/s123600g/asr_edgetpu_demo/blob/master/images/2020-02-16%2020-03-00%20%E7%9A%84%E8%9E%A2%E5%B9%95%E6%93%B7%E5%9C%96.png)

並且在終端機上會看到一則訊息為：
```
可在終端機使用下列命令，來進行EdgeTPU Compiler TFLite model
>> edgetpu_compiler -s /home/jyu/Program/Audio_Speech_Recognition_TPU_Demo/tflite_model/ASR_Model.tflite <<
```
複製顯示命令並貼在終端機上執行<br/>
>  edgetpu_compiler -s /home/jyu/Program/Audio_Speech_Recognition_TPU_Demo/tflite_model/ASR_Model.tflite

![image](https://github.com/s123600g/asr_edgetpu_demo/blob/master/images/2020-02-16%2020-06-30%20%E7%9A%84%E8%9E%A2%E5%B9%95%E6%93%B7%E5%9C%96.png)

最後編譯轉換完成tflite模型存放在tflite_model/ASR_Model.tflite <br/>

**Step 6. 使用 Accelerator 進行分類預測**<br/>
請將 Google Coral Accelerator 插入USB，執行classify_ASR.py。

>  python classify_ASR.py

![image](https://github.com/s123600g/asr_edgetpu_demo/blob/master/images/2020-02-16%2019-58-36%20%E7%9A%84%E8%9E%A2%E5%B9%95%E6%93%B7%E5%9C%96.png)

