# -*- coding: utf-8 -*-

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



http://noahsnail.com/2017/04/29/2017-4-29-matplotlib%E7%9A%84%E5%9F%BA%E6%9C%AC%E7%94%A8%E6%B3%95(%E5%9B%9B)%E2%80%94%E2%80%94%E8%AE%BE%E7%BD%AElegend%E5%9B%BE%E4%BE%8B/
https://blog.csdn.net/Quincuntial/article/details/70947363
https://www.jianshu.com/p/91eb0d616adb
'''

import matplotlib.pyplot as plt
import os


def plot_figure(history, dir, figure_classname):

    __plot_acc(history, dir, figure_classname)

    # plt.subplot(1,2,1)

    __plot_loss(history, dir, figure_classname)

    # plt.show()


def __plot_acc(hist, path, figure_classname):
    '''
    輸出模型訓練過程之準確度變化圖
    '''

    acc = hist.history['acc']
    val_acc = hist.history['val_acc']

    # figure size
    plt.figure(figsize=(8, 8))
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title(figure_classname+'_Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train_acc', 'valid_acc'], loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(path, str(figure_classname+'_Accuracy.png')))
    # plt.show()


def __plot_loss(hist, path, figure_classname):
    '''
    輸出模型訓練過程之損失函數變化圖
    '''

    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    # figure size
    plt.figure(figsize=(8, 8))
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title(figure_classname+'_Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train_loss', 'valid_loss'], loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(path, str(figure_classname+'_Loss.png')))
    # plt.show()
