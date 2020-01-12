# -*- coding:utf-8 -*-

import librosa
import librosa.display
import os
import numpy as np
import matplotlib.pyplot as plt

sample_rate = 16000
channel = 1  # 單一通道(single channel)
Audio_Data_DirectoryName = "prediction_data"
Audio_Data_Path = os.path.join(
    os.getcwd(),
    Audio_Data_DirectoryName
)

Save_Img_Root_DirectoryName = "plot_figure"
Save_Img_DirectoryName = "audio_spectrum"
Save_Img_Path = os.path.join(
    os.getcwd(),
    Save_Img_Root_DirectoryName,
    Save_Img_DirectoryName
)


if __name__ == "__main__":

    # try:

    if not os.path.exists(Audio_Data_Path):  # 判斷音檔存放位置是否存在

        raise Exception(
            "The ['{}'] is  not exists !!".format(Audio_Data_Path)
        )

    elif not os.path.exists(Save_Img_Path):  # 判斷輸出圖片存放位置是否存在

        raise Exception(
            "The ['{}'] is  not exists !!".format(Save_Img_Path)
        )

    else:

        print("[plot_audio] Start load audio data....\n")

        ''' 判斷放置紀錄檔案目錄位置內是否存在有目錄 '''
        if os.listdir(Save_Img_Path) != 0:

            ''' 讀取放置紀錄檔案目錄位置內容 '''
            for read_dir in os.listdir(Save_Img_Path):

                ''' 判斷當前讀取放置紀錄檔案目錄位置內容項目是否為目錄 '''
                if os.path.isdir(os.path.join(Save_Img_Path, read_dir)):

                    ''' 讀取當前放置紀錄檔案目錄位置內容目錄內檔案 '''
                    for read_content in os.listdir(os.path.join(Save_Img_Path, read_dir)):

                        ''' 刪除當前讀取到檔案 '''
                        os.remove(
                            os.path.join(
                                Save_Img_Path,
                                read_dir,
                                # 當前讀取放置紀錄檔案目錄位置內容目錄內檔案
                                read_content
                            )
                        )

                    ''' 刪除當前目錄 '''
                    os.rmdir(
                        os.path.join(
                            Save_Img_Path,
                            read_dir,
                        )
                    )

        ''' 讀取當前指定位置內容 '''
        for read_dir in os.listdir(Audio_Data_Path):

            ''' 判斷當前的路徑是否為一個目錄 '''
            if os.path.isdir(os.path.join(Audio_Data_Path, read_dir)):

                print(
                    "[plot_audio] Current Load Directory Name ['{}']".format(
                        read_dir)
                )

                ''' 建立當前類別目錄 '''
                os.mkdir(
                    os.path.join(
                        Save_Img_Path,
                        read_dir
                    )
                )

                ''' 掃描檔案 '''
                for read_file in os.listdir(os.path.join(Audio_Data_Path, read_dir)):

                    print(
                        "[plot_audio] Current Load Directory Name ['{}'] -->  File ['{}']".format(
                            read_dir,
                            read_file
                        )
                    )

                    ''' 讀取音頻檔案 '''
                    wave, sr = librosa.load(os.path.join(
                        Audio_Data_Path,
                        read_dir,
                        read_file
                    ),
                        duration=5.0,
                        mono=True,
                        sr=sample_rate
                    )

                    ''' 設置圖片大小 '''
                    plt.figure(figsize=(12, 8))

                    # plt.subplot(4, 2, 7)

                    data = librosa.amplitude_to_db(
                        np.abs(librosa.stft(wave)), ref=np.max)

                    ''' 產生音頻頻譜圖 '''
                    librosa.display.specshow(
                        data=data, sr=sample_rate, x_axis='time', y_axis='log')

                    plt.colorbar(format='%+2.0f dB')
                    plt.title('Log power spectrogram')

                    ''' 音頻頻譜圖存檔 '''
                    plt.savefig(os.path.join(Save_Img_Path,
                                             read_dir, str(read_file+'.png')))

                print()

    # except Exception as err:

    #     print("\n>>> {} <<<".format(err))
