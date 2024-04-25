# -*- coding: utf-8 -*-

#%% CV, continuous data, 데이터명이나 경로는 islab server내 도커 컨테이너에 맞춰져 있음
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from itertools import product
from sklearn.tree import DecisionTreeClassifier as skDTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import timeit

datapath = '/home'

cv = 3
#dset = ['cancer', 'digit', 'iris', 'wine']

#dset = ['Adult', 'BreastCancer', 'Connect-4', 'Soybean']
dset = ['cancer', 'cover-small', 'digit', 'iris', 'wine']+['cancer']*4 + ['cover-small']*4 + ['digit']*4 + ['iris']*4 + ['wine']*4
dset1 = ['None']*5 + ['doane','fd','scott','sturges']*5

tree_model = 'Binary'
depth_set = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
n_ests = [1, ]

os.chdir(datapath)

enc = LabelEncoder()


result = pd.DataFrame(
    columns=['data', 'binning','n_est', 'criterion', 'max_feat', 'depth', 'fold', 'total_time', 'avg_time', 'std_time'])
for dname, dname1 in zip(dset, dset1):
    if dname1 == 'None':
        data = pd.read_csv('%s.csv' % (dname))
    else:
        data = pd.read_csv('%s_%s.csv' % (dname, dname1))
    cv_ind = pd.read_csv('%s_fold.csv' % (dname))

    X = data.drop('label', axis=1)
    y = data['label']

    for max_depth, criterion, num_estimators in product(depth_set, ['mgini',], n_ests):

        for i in range(cv):
            print(dname, dname1, num_estimators, max_depth, criterion, i)
            if num_estimators == 1:
                clf = skDTreeClassifier(max_depth=max_depth, criterion=criterion, max_features=None)
            else:
                clf = RandomForestClassifier(max_depth=max_depth, criterion=criterion,
                                             max_features=None, n_estimators=num_estimators)

            trn_ind = cv_ind[cv_ind['test_fold'] != i]['ind'].values
            val_ind = cv_ind[cv_ind['test_fold'] == i]['ind'].values
            t1 = time.time()
            clf.fit(X.loc[trn_ind], y.loc[trn_ind])
            tr_time = time.time() - t1
            def predict_func():
                y_pred = clf.predict(X.loc[val_ind])
    
            # 1000번씩 실행하는 것을 100번 반복
            repeat_num = 100
            times = timeit.repeat(predict_func, number=1000, repeat=repeat_num)
            
            # 총합, 평균 및 표준편차 계산
            total_time = np.sum(times)
            avg_time = np.mean(times)
            std_time = np.std(times)
        
            # 결과 저장
            result.loc[len(result)] = [dname, dname1, num_estimators, criterion, 'None',
                                       max_depth, i, total_time, avg_time, std_time]
    print('%s complete'% (dname))
result.to_csv('result_summary_conti_%s_week3.csv' % (tree_model), index=False)

