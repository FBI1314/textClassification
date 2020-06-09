## Project Name

  文本多标签分类

## Project Introduce

​  实现封装SGM、Bert+SGM、Bert+Seq2set、Bert+CNN多标签分类模型

## project Tree

|--config  
|　　　|--basic_config.py   　　　#模型与训练参数的配置文件  
|--predata  
|　　　|--preprocess.py     　　　#数据预处理方法   
|--train  
|　　　|--losses.py     　　　#损失函数类  
|　　　|--metrics.py    　　　#评价方法类  
|　　　|--trainer.py    　　　#训练程序  
|--models  
|　　　|--optims.py     　　　#优化方法类  
|　　　|--sgm_encoder_decoder.py  　　　#sgm_生成模型encoder、decoder 
|　　　|--seq2set_encoder_decoder.py  　　　#seq2set_生成模型encoder、decoder 
|　　　|--attention.py      　　　#注意力方法  
|　　　|--model.py          　　　#模型构造类  
|　　　|--beam.py           　　　#Beamsearch 工具类  
|　　　|--textcnn.py        　　　# TextCNN定义类  
|　　　|--sgm.py            　　　# SGM、BertSGM模型定义类 
|　　　|--seq2set.py            　　　# Bertseq2set模型定义类 
|　　　|--bert_for_multi_label.py 　　　#BertCNN 模型定义类  
|--data  
|　　　|--save_data        　　　#数据预处理存储目录  
|　　　|--test.tgt         　　　#测试集label文件  
|　　　|--test.src         　　　#测试集input文件  
|　　　|--valid.src        　　　#验证集input文件         
|　　　|--valid.tgt        　　　#验证集label文件  
|　　　|--train.src        　　　#训练集input文件  
|　　　|--train.tgt        　　　#训练集label文件  
|　　　|--labels.txt       　　　#所有label  
|　　　|--label_sorted.json 　　　 #lable数目排序文件  
|--main.py               　　　 #主程序  
|--utils  
|　　　|--utils.py       　　　    #工具类  
|　　　|--dict_helper.py 　　　 #字段工具类  
|--result                　　　 #结果存储目录  
|--bert_pretrain 　　　 #预训练模型存储目录  
|　　　|--vocab.txt  
|　　　|--config.json  
|　　　|--pytorch_model.bin  

##  Using the step

   1.参考data目录准备带训练数据  
   2.根据需求修改config/basic_config.py文件中的训练参数  
   3.运行main.py文件

##  论文
   SGM: https://arxiv.org/abs/1806.04822
   Seq2Set:https://arxiv.org/pdf/1809.03118.pdf

## Version
  python3.6
  pytorch

## **Environment**

  pip install -r requirements.txt

## **run**

  python3 main.py

 

