# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 09:51:17 2022

@author: Александр
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import time

class Ansamble(object):
    def __init__(self):
        self.__data = datasets.load_wine()
        self.__X = self.__data.data
        self.__y = self.__data.target
        
    def createSample(self, test_size):
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(self.__X, self.__y,test_size=test_size)   
    
    def createTree(self):
        self.__CNN = DecisionTreeClassifier()
        
    def createGaussian(self):
        self.__CNN = GaussianNB()
        
    def createLR(self):
        self.__CNN = LinearRegression()
        
    def createKNN(self):
        self.__CNN = KNeighborsClassifier(n_neighbors=7)
        
    def createBusting(self):
        self.__CNN = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
    
    def createBagging(self):
        self.__CNN = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
        
    def createStacking(self):
        log = GaussianNB()
        tree = DecisionTreeClassifier()
        logreg = KNeighborsClassifier(n_neighbors=7)
        self.__CNN = VotingClassifier(estimators=[('lr', log), ('knn', logreg), ('tree', tree)], voting='hard')
        
    def makePrediction(self):
        self.__CNN.fit(self.__X_train,self.__y_train)
        return self.__CNN.score(self.__X_test,self.__y_test)
    
    methods = {'Tree    ': createTree,'Gaussian':createGaussian, 'LR      ':createLR,
               'KNN     ':createKNN, 'Busting ':createBusting, 'Bagging ': createBagging,
               'Stacking':createStacking}   
        
k=100
res=np.zeros((7,k))
test1=Ansamble();
for i,j in enumerate(test1.methods):
    start_time = time.time()
    for s in range(k):
        test1.createSample(0.8)
        test1.methods[j](test1)
        res[i][s]=test1.makePrediction()
    time1 = time.time() - start_time
    print('method:{} accuracy: {:5f}  time: {:8f}'.format(j,res[i].mean(),time1))