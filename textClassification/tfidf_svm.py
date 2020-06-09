#!/usr/bin/env python
# encoding: utf-8
'''
@author: fangbing
@contact: fangbing@cvte.com
@file: tfidf_svm.py
@time: 2019/2/11 19:26
@desc:
'''
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
import numpy as np


#得到准确率和召回率
def evaluate(actual, pred):
    m_precision = metrics.precision_score(actual, pred,average='macro')
    m_recall = metrics.recall_score(actual,pred,average='macro')
    m_acc=metrics.accuracy_score(actual,pred)
    f1=(2*(m_precision*m_recall))/(m_precision+m_recall)
    print('precision:{0:.3f}'.format(m_precision))
    print('recall:{0:0.3f}'.format(m_recall))
    print('f1:{0:0.3f}'.format(f1))
    print('acc:{0:0.3f}'.format(m_acc))

def readData():
    f=open('easyCNNData.txt','r',encoding='utf-8')
    lines1=f.readlines()
    f2=open('easyCNNLabel.txt','r',encoding='utf-8')
    lines2=f2.readlines()
    print(len(lines1))
    print(len(lines2))

    trainData=[]
    testData=[]
    trainLabel=[]
    testLabel=[]
    count=0
    for text,label in zip(lines1,lines2):
        count+=1
        if(count<50000):
            testData.append(text.replace('\n',''))
            testLabel.append(label.replace('\n',''))
        else:
            trainData.append(text.replace('\n',''))
            trainLabel.append(label.replace('\n',''))

    return trainData,testData,trainLabel,testLabel

def tfidfvectorize(train_words,test_words):
    v = TfidfVectorizer(binary = False, decode_error = 'ignore',stop_words = 'english')
    train_data = v.fit_transform(train_words)
    test_data = v.transform(test_words)
    return train_data,test_data

#创建svm分类器
def train_clf(train_data, train_tags):
    clf = SVC(C=1.0, cache_size=50, class_weight=None, coef0=0.0,
                  decision_function_shape=None, degree=3, gamma='auto',
                  kernel='rbf', max_iter=-1, probability=False, random_state=None,
                  shrinking=True, tol=0.001, verbose=False)
    clf.fit(train_data, np.asarray(train_tags))
    return clf


from scipy import sparse
if __name__=='__main__':
    trainData, testData, trainLabel, testLabel=readData()
    train_data, test_data=tfidfvectorize(trainData, testData)

    print(type(train_data))
    print(type(trainLabel))
    clf=train_clf(train_data, trainLabel)
    re = clf.predict(test_data)
    print(re)
    evaluate(np.asarray(testLabel), re)
    print(re)
