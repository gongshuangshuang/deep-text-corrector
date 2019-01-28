1.数据集：

  Cornell电影对话数据，由30,4713行电影脚本组成.

2.数据处理：

  首先进行数据处理，提取出对话文本，并将数据集切分成训练集、验证集、测试集，对应的数据量分别为24,3770行，3,0471行， 3,0472行.
  
  命令行：python data_preprocess.py

3.功能函数注解：
  data_reader.py:一个抽象类，它为能够读取源数据集并生成输入-输出对的类定义接口，其中输入是源语句语法错误的变体，输出是原始语句.
  
  text_corrector_data_readers.py:包含一些DataReader的实现，在Cornell电影对话框语料库上.
  
  text_corrector_models.py:包含一个修改过的seq2seqmodel版本，以实现偏向解码中描述的逻辑.
  
  correct_text.py：一组辅助函数，包含对模型进行训练和测试。
  
 4.


    





