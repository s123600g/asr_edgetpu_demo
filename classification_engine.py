# -*- coding:utf-8 -*-

from edgetpu.basic.basic_engine import BasicEngine
import numpy


class ClassificationEngine(BasicEngine):
    """Engine used for classification task."""

    def __init__(self, model_path, device_path=None):
        """
        建立一個基於指定模型BasicEngine類別實體。

        Args:
          model_path: TF-Lite 模型檔案來源位置，格式為字串型態。
          device_path: 預設為None代表使用預設系統抓到Edge TPU裝置，如果有指定使用哪一個Edge TPU裝置，在執行時會使用指定Edge TPU裝置，
          如果有需要指定需要給予格式為字串型態。

        Raises:
          ValueError: An error occurred when the output format of model is invalid.
        """

        ''' 判斷是否有指定Edge TPU裝置 '''
        if device_path:
            super().__init__(model_path, device_path)

        else:

            super().__init__(model_path)

        ''' 取得模型輸入層張量型態 '''
        input_tensor_shape = self.get_input_tensor_shape()

        ''' 取得模型輸出層張量大小 '''
        get_all_output_tensors_sizes = self.get_all_output_tensors_sizes()

        ''' 取得模型輸出層個數 '''
        get_num_of_output_tensors = self.get_num_of_output_tensors()

        ''' 取得指定模型輸出層(透過索引)之大小 '''
        get_output_tensor_size = self.get_output_tensor_size(0)

        ''' 取得模型之資料輸入大小要求 '''
        required_input_array_size = self.required_input_array_size()

        print("[classification_engine] get_all_output_tensors_sizes：{}".format(
            get_all_output_tensors_sizes
        ))

        print("[classification_engine] get_num_of_output_tensors：{}".format(
            get_num_of_output_tensors
        ))

        print("[classification_engine] get_output_tensor_size：{}".format(
            get_output_tensor_size
        ))

        print("[classification_engine] input_tensor_shape：{}".format(
            input_tensor_shape
        ))

        print("[classification_engine] required_input_array_size：{}".format(
            required_input_array_size
        ))

        ''' 判斷模型輸出層張量大小是否不為1  '''
        if get_all_output_tensors_sizes.size != 1:
            raise ValueError(
                ('Classification model should have 1 output tensor only!'
                 'This model has {}.'.format(get_all_output_tensors_sizes.size)))

    def ClassifyWithASR_Feature(
            self, Fdata, threshold=0.1, top_k=3):
        """
        這是載入已分類訓練好模型進行推斷預測方法。

        Args:
          Fdata: 輸入用來分類預測之資料(特徵值)。
          threshold: 設定一個小數(float)作為篩選門檻，從可信度清單內過濾每一個數值是否有達到門檻(等於和大於)，預設值為0.1。
          top_k: 取得指定數量最佳可信度，所謂最佳可信度數量指的是從模型預測出來結果中，經過排序之後，
          根據排序結果取得指定數量之最大可信度(由大至小)，預設為3代表要取出3個最大可信度。

        Returns:
          一組包含索引和評分之清單結果(int, float)。

        Raises:
          RuntimeError: when model isn't used for image classification.
        """

        ''' 取得模型之輸入層格式大小 '''
        input_tensor_shape = self.get_input_tensor_shape()

        ''' 判斷取得模型之輸入層格式大小是否維度不等於4與第一個位置參數是否不為1，代表要求模型輸入層格式大小不對 '''
        if (input_tensor_shape.size != 4 or input_tensor_shape[0] != 1):
            raise RuntimeError(
                'Invalid input tensor shape! Expected: [1, height, width, 1]')

        print("[ClassificationEngine] input_tensor_shape： {}".format(
            input_tensor_shape
        ))

        # print("[ClassificationEngine] Fdata{}".format(Fdata))

        print("[ClassificationEngine] numpy img shape： {}".format(
            numpy.asarray(Fdata).shape
        ))
        # print("[ClassificationEngine] numpy img data： {}".format(
        #     numpy.asarray(Fdata)
        # ))

        ''' 
        特徵資料進行平坦化，符合模型輸入層之大小
        假設此模型輸入層要求大小為220，在我們輸入的特徵資料格式(20,11)--> 20*11 = 220
        '''
        input_tensor = numpy.asarray(Fdata).flatten()
        # print("[ClassificationEngine] input_tensor： {}".format(input_tensor))

        return self.ClassifyWithInputTensor(input_tensor, threshold, top_k)

    def ClassifyWithInputTensor(self, input_tensor, threshold=0.0, top_k=3):
        """
        將一組資料張量進行分類推斷預測方法

        此方法為根據使用者提交的一組資料張量進行推斷預測，使用者提交之資料張量，必須是經過轉換後符合模型所要求之格式。

        Args:
          input_tensor: 一組用來提交給輸入層之numpy array型態資料。
          threshold: 設定一個小數(float)作為篩選門檻，從可信度清單內過濾每一個數值是否有達到門檻(等於和大於),預設值為0.0。
          top_k: 指定取得最佳可信度數量，所謂最佳可信度數量指的是從模型預測出來結果中，最後排序取得最大可信度結果清單，
          預設為3代表要取出3個最大可信度。

        Returns:
          一組包含索引和評分之清單結果(int, float)。

        Raises:
          ValueError: when input param is invalid.
        """
        if top_k <= 0:
            raise ValueError('top_k must be positive!')

        ''' 進行模型推斷預測並取得預測結果 '''
        latency_time, self._raw_result = self.RunInference(
            input_tensor)

        # print("[ClassificationEngine] get_raw_output：{}".format(self.get_raw_output()))

        print("[ClassificationEngine] latency_time：{:.2f}ms".format(
            latency_time))

        # print("[ClassificationEngine] {:.2f}ms".format(
        #     self.get_inference_time()))

        # top_k must be less or equal to number of possible results.
        top_k = min(top_k, len(self._raw_result))

        print("[ClassificationEngine] top_k：{}".format(top_k))

        print("[ClassificationEngine] raw_result type：{} , raw_result size：{} ,  raw_result shape：{} ".format(
            type(self._raw_result),
            self._raw_result.size,
            self._raw_result.shape
        ))

        print("[ClassificationEngine] raw_result：\n{}".format(
            self._raw_result
        ))

        result = []

        '''排序模型回傳最大可信度清單 '''
        order_indices = numpy.argpartition(self._raw_result, -top_k)

        print("[ClassificationEngine] order_indices：{}".format(
            order_indices
        ))

        print("[ClassificationEngine] order_indices result：{}".format(
            self._raw_result[order_indices]
        ))

        ''' 
        排序模型回傳最大可信度清單，抓出指定數量之最大可信度，抓出來的結果會由最小排到最大，內容為抓出指定數量之最大結果索引位置清單。

        numpy.argpartition(array, kth)[-top_k:]
        Args：
        array:要排序之Array來源。
        kth：指定Array來源內部索引位置，將此索引位置內數值作為參考依據，進行內部篩選排序，只要小於此參考數值就會往此索引位置前面放置排序，
        如果大於或等於此參考數值就會往此索引位置後面放置排序。
        -top_k:指定從Array後面取出結果數量。

        '''
        indices = numpy.argpartition(self._raw_result, -top_k)[-top_k:]

        print("[ClassificationEngine] indices：{}".format(indices))

        ''' 讀取排序後放置索引位置清單 '''
        for i in indices:

            ''' 判斷該索引位置數值(可信度)是否有達到門檻，大於門檻才符合條件 '''
            if self._raw_result[i] >threshold:

                ''' 將索引與可信度加入結果清單 '''
                result.append((i, self._raw_result[i]))

        # print("[ClassificationEngine] result：{}".format(result))

        ''' 進行篩選結果排序，反轉排序為由大到小 '''
        result.sort(key=lambda tup: -tup[1])

        print("[ClassificationEngine] result：{}".format(result))

        '''
        假設result排序結果如下

           0     1    2 
         *---*---*---*
         | 23| 24| 5 |
         *---*---*---*
          -2    -1    -0
        
        result[:top_k]就是取得範圍從 start ~ (top_k-1)

        '''

        return result[:top_k]
