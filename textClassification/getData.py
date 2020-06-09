#!/usr/bin/env python
# encoding: utf-8
'''
@author: fangbing
@contact: fangbing@cvte.com
@file: cleanData.py
@time: 2019/1/30 14:20
@desc:
'''
import pymysql
import re
import pandas as pd
import jieba
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE # 过抽样处理库SMOTE
from imblearn.ensemble import EasyEnsemble
import pickle
import sys

db2=pymysql.connect(host='****',port=3306,user='***',passwd='***',db='***')
cur2=db2.cursor()

# 欠采样
def undersample(data):
    x = data.iloc[:, 0:2]
    y = data.iloc[:, -2]
    print(x.count())
    print('===========')
    print(y.count())
    model_RandomUnderSampler = RandomUnderSampler()
    x_RandomUnderSampler_resampled, y_RandomUnderSampler_resampled = model_RandomUnderSampler.fit_sample(x,y)  # 输入数据并作欠抽样处理
    x_RandomUnderSampler_resampled = pd.DataFrame(x_RandomUnderSampler_resampled, columns=['index', 'question'])
    y_RandomUnderSampler_resampled = pd.DataFrame(y_RandomUnderSampler_resampled,
                                                  columns=['tag_name'])  # 将数据转换为数据框并命名列名
    RandomUnderSampler_resampled = pd.concat([x_RandomUnderSampler_resampled, y_RandomUnderSampler_resampled],
                                             axis=1)  # 按列合并数据框
    groupby_data_RandomUnderSampler = RandomUnderSampler_resampled.groupby('tag_name').count()  # 对label做分类汇总
    print(groupby_data_RandomUnderSampler)  # 打印输出经过RandomUnderSampler处理后的数据集样本分类分布
    return RandomUnderSampler_resampled

# 使用SMOTE方法进行过抽样处理
def upsample(data):

    # labels = list(data['tag_name'])
    # labelsDict2 = {}
    # for index, lb in enumerate(list(set(labels))):
    #     labelsDict2[str(lb)] = index
    # print(labelsDict2)
    # data['tag_name'] = data['tag_name'].map(labelsDict2)
    # print(data['tag_name'])

    x = data.iloc[:, 0:2]
    y = data.iloc[:, -2]
    # print(data.groupby('tag_name').count())
    print('-------------------------------------------------')
    print(x.count())
    print('===========')
    print(y.count())
    model_smote = SMOTE()  # 建立SMOTE模型对象
    x_smote_resampled, y_smote_resampled = model_smote.fit_sample(x,y)  # 输入数据并作过抽样处理
    x_smote_resampled = pd.DataFrame(x_smote_resampled,columns=['index', 'question'])  # 将数据转换为数据框并命名列名
    y_smote_resampled = pd.DataFrame(y_smote_resampled, columns=['tag_name'])  # 将数据转换为数据框并命名列名
    smote_resampled = pd.concat([x_smote_resampled, y_smote_resampled], axis=1)  # 按列合并数据框
    groupby_data_smote = smote_resampled.groupby('tag_name').count()  # 对label做分类汇总
    print(groupby_data_smote)  # 打印输出经过SMOTE处理后的数据集样本分类分布

def myUpsample(data):

    labels = list(data['tag_name'])
    labelsDict2 = {}
    for index, lb in enumerate(list(set(labels))):
        labelsDict2[str(lb)] = index
    print(labelsDict2)
    data['tag_name'] = data['tag_name'].map(labelsDict2)


    data = data.sample(frac=1).reset_index(drop=True)
    datagp=data.groupby('tag_name')
    # 设定样本数
    sample_num = 40000
    dataSample=list()
    testSample=list()
    for name,gp in datagp:
        print(name,len(gp['question']))

        if (type(testSample)==list):
            print(type(gp))
            testSample=gp.iloc[:30,:]
        else:
            testSample = pd.concat([testSample, gp.iloc[:30,:]], ignore_index=True)

        gp = gp.iloc[30:,:].sample(sample_num, replace=True)
        print(name, len(gp['question']))
        if (type(dataSample)==list):
            dataSample=gp
        else:
            dataSample = pd.concat([dataSample, gp], ignore_index=True)
    print(dataSample.groupby('tag_name').count())
    print(testSample.groupby('tag_name').count())
    return dataSample,testSample


def EasySample(data):
    x = data.iloc[:, 0:2]
    y = data.iloc[:, -2]
    # 使用集成方法EasyEnsemble处理不均衡样本
    model_EasyEnsemble = EasyEnsemble()  # 建立EasyEnsemble模型对象
    x_EasyEnsemble_resampled, y_EasyEnsemble_resampled =model_EasyEnsemble.fit_sample(x, y)  # 输入数据并应用集成方法处理
    print(x_EasyEnsemble_resampled.shape)  # 打印输出集成方法处理后的x样本集概况
    print(y_EasyEnsemble_resampled.shape)  # 打印输出集成方法处理后的y标签集概况


jieba.load_userdict('../preData/mid_math_dict.txt')

def loadstopWords(file_path):
    stopwords = [line.strip() for line in open(file_path, 'r', encoding='utf-8')]
    return stopwords
stopwords = loadstopWords('../preData/ha_stopwords.txt')
def jieba_cut(sentence):
    sents = jieba.cut(sentence.strip())
    result = []
    for i in sents:  ##去除停用词表中的词；
        if i not in stopwords:
            result.append(i)
    result = ' '.join(result)
    return result

def getDataforBert():
    import random
    data = pd.read_csv('../preData/shuxueabilityQuestion.csv')
    data = data.dropna(axis=0, how='any')
    RandomUndersampled=undersample(data)

    # RandomUndersampled = upsample(data)
    #
    # questions = list(RandomUndersampled['question'])
    # labels = list(RandomUndersampled['tag_name'])

    questions = list(RandomUndersampled['question'])
    labels = list(RandomUndersampled['tag_name'])

    testData=data.loc[~data['question'].isin(questions)]
    testq=list(testData['question'])[:100]
    testlb=list(testData['tag_name'])[:100]

    labelsDict2 = {}
    for index, lb in enumerate(list(set(labels))):
        labelsDict2[str(lb)] = index
    print(labelsDict2)

    f = open('trainData_bert.txt', 'a', encoding='utf-8')

    f2 = open('devData_bert.txt', 'a', encoding='utf-8')

    f3 = open('testData_bert.txt', 'a', encoding='utf-8')

    train=[]
    dev=[]
    test=[]
    testDict = {}
    for text, label in zip(questions, labels):
        text = jieba_cut(text)
        label = str(labelsDict2.get(label))
        if (testDict.get(label) is None or len(testDict.get(label)) < 30):
            if(testDict.get(label) is None):
                testDict[label] = [label + '\t' + str(text) + '\n']
            else:
                testDict[label].append(label + '\t' + str(text) + '\n')
            # f2.write(label + '\t' + text + '\n')
            dev.append(label + '\t' + text + '\n')
            continue
        # f.write(label + '\t' + text + '\n')
        train.append(label + '\t' + text + '\n')

    for text, label in zip(testq, testlb):
        text = jieba_cut(text)
        label = str(labelsDict2.get(label))
        # f3.write(label + '\t' + text + '\n')
        test.append(label + '\t' + text + '\n')
    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)

    for t in train:
        f.write(t)
    for d in dev:
        f2.write(d)
    for t in test:
        f3.write(t)

def getDataforSVM():
        data = pd.read_csv('../preData/shuxueabilityQuestion.csv')
        data = data.dropna(axis=0, how='any')
        RandomUndersampled = undersample(data)
        # RandomUndersampled = upsample(data)
        #
        # questions = list(RandomUndersampled['question'])
        # labels = list(RandomUndersampled['tag_name'])


        questions = list(RandomUndersampled['question'])
        print(questions)
        labels = list(RandomUndersampled['tag_name'])

        labelsDict2 = {}
        for index, lb in enumerate(list(set(labels))):
            labelsDict2[str(lb)] = index
        print(labelsDict2)
        # sys.exit(0)
        all_datas = []
        all_labels = []
        test_datas = []
        test_labels = []
        testDict = {}
        for text, label in zip(questions, labels):
            text = jieba_cut(text)
            label = str(labelsDict2.get(label))

            if (testDict.get(label) is None or len(testDict.get(label)) < 50):
                if (testDict.get(label) is None):
                    testDict[label] = [label + '\t' + str(text) + '\n']
                else:
                    testDict[label].append(label + '\t' + str(text) + '\n')
                test_datas.append(text)
                test_labels.append(label)
                continue
            all_datas.append(text)
            all_labels.append(label)

        print(len(all_datas))
        print(len(all_labels))
        print(len(test_datas))
        print(len(test_labels))

        pickle.dump(all_datas, open('under_trainData.pkl', 'wb'))
        pickle.dump(all_labels, open('under_trainLabel.pkl', 'wb'))

        pickle.dump(test_datas, open('under_testData.pkl', 'wb'))
        pickle.dump(test_labels, open('under_testLabel.pkl', 'wb'))


def getDataforCNN():
    data = pd.read_csv('../preData/shuxueabilityQuestion.csv')
    data = data.dropna(axis=0, how='any')
    # RandomUndersampled = undersample(data)
    RandomUndersampled = myUpsample(data)

    print(type(RandomUndersampled))
    print(RandomUndersampled)

    # RandomUndersampled=EasySample(data)
    # RandomUndersampled = upsample(data)
    #
    # questions = list(RandomUndersampled['question'])
    # labels = list(RandomUndersampled['tag_name'])
    print(list(RandomUndersampled['question'])[:5])
    RandomUndersampled=RandomUndersampled.sample(frac=1).reset_index(drop=True)
    print(list(RandomUndersampled['question'])[:5])
    questions = list(RandomUndersampled['question'])
    labels = list(RandomUndersampled['tag_name'])

    labelsDict2 = {}
    for index, lb in enumerate(list(set(labels))):
        labelsDict2[str(lb)] = index
    print(labelsDict2)

    f=open('trainData_cnn.txt','a',encoding='utf-8')
    f2=open('labelData_cnn.txt','a',encoding='utf-8')
    for text, label in zip(questions, labels):
        text = jieba_cut(text)
        label = str(labelsDict2.get(label))
        # f.write(text+'\n')
        # f2.write(str(label)+'\n')
    f.close()
    f2.close()


def getMyUpSample():
    data = pd.read_csv('../preData/shuxueabilityQuestion.csv')
    data = data.dropna(axis=0, how='any')


    myTrainSample, myTest = myUpsample(data)
    train_questions = list(myTrainSample['question'])
    train_labels = list(myTrainSample['tag_name'])

    test_questions = list(myTest['question'])
    test_labels = list(myTest['tag_name'])

    trainDatas = []
    trainLabels = []
    testDatas = []
    testLabels = []

    for text, label in zip(train_questions, train_labels):
        text = jieba_cut(text)
        trainDatas.append(text)
        trainLabels.append(label)

    for text, label in zip(test_questions, test_labels):
        text = jieba_cut(text)
        testDatas.append(text)
        testLabels.append(label)


    print(len(trainDatas))
    print(len(trainLabels))
    print(len(testDatas))
    print(len(testLabels))

    pickle.dump(trainDatas, open('up_trainData.pkl', 'wb'))
    pickle.dump(trainLabels, open('up_trainLabel.pkl', 'wb'))

    pickle.dump(testDatas, open('up_testData.pkl', 'wb'))
    pickle.dump(testLabels, open('up_testLabel.pkl', 'wb'))

def getUpSampleDataforBert():
    import random
    data = pd.read_csv('../preData/shuxueabilityQuestion.csv')
    data = data.dropna(axis=0, how='any')

    myTrainSample, myTest = myUpsample(data)
    train_questions = list(myTrainSample['question'])
    train_labels = list(myTrainSample['tag_name'])

    test_questions = list(myTest['question'])
    test_labels = list(myTest['tag_name'])

    f = open('up_trainData_bert.txt', 'a', encoding='utf-8')

    # f2 = open('up_devData_bert.txt', 'a', encoding='utf-8')

    f3 = open('up_devData_bert.txt', 'a', encoding='utf-8')

    train=[]
    # dev=[]
    test=[]
    testDict = {}
    for text, label in zip(train_questions, train_labels):
        text = jieba_cut(text)
        # if (testDict.get(str(label)) is None or len(testDict.get(str(label))) < 300):
        #     if(testDict.get(str(label)) is None):
        #         testDict[str(label)] = [str(label) + '\t' + str(text) + '\n']
        #     else:
        #         testDict[str(label)].append(str(label) + '\t' + str(text) + '\n')
        #     # f2.write(label + '\t' + text + '\n')
        #     dev.append(str(label) + '\t' + text + '\n')
        #     continue
        # f.write(label + '\t' + text + '\n')
        train.append(str(label) + '\t' + text + '\n')

    for text, label in zip(test_questions, test_labels):
        text = jieba_cut(text)
        # f3.write(label + '\t' + text + '\n')
        test.append(str(label) + '\t' + text + '\n')

    print(len(train))
    # print(len(dev))
    print(len(test))
    random.shuffle(train)
    # random.shuffle(dev)
    random.shuffle(test)

    for t in train:
        f.write(t)
    # for d in dev:
    #     f2.write(d)
    for t in test:
        f3.write(t)


# getDataforBert()
# getDataforCNN()
# getMyUpSample()
getDataforSVM()
getUpSampleDataforBert()


