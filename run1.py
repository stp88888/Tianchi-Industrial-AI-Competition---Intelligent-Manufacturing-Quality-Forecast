"""
Created on Mon Dec 11 10:08:14 2017

@author: STP
"""

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from collections import defaultdict
import copy
import time
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, ShuffleSplit
#/from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import minmax_scale
import scipy as sc

def calc_same(x, y):
    data = copy.deepcopy(y)
    feature_list = copy.deepcopy(x)
    columns = data.columns
    columns_dictionary = defaultdict(lambda: 0)
    k = 0
    for i in columns:
        columns_dictionary[k] = i
        k += 1

    data = data.values
    feature_drop = []
    huancun = []
    k = 0
    print('total:%s' % (len(feature_list)))
    for i in range(data.shape[1]):
        if k % 1000 == 0:
            print('current:%s' % (k))
        for j in range(data.shape[1]):
            if i != j:
                if (data[:, i] == data[:, j]).all():
                    if str(j).zfill(4) + str(i).zfill(4) not in huancun:
                        feature_drop.append([i, j])
                        huancun.append(str(i).zfill(4) + str(j).zfill(4))
                        huancun.append(str(j).zfill(4) + str(i).zfill(4))
        k += 1
    return feature_drop, columns_dictionary


def calc_append(x, y):
    same = copy.deepcopy(x)
    feature = copy.deepcopy(y)
    same.append(feature)
    same2 = []
    for i in range(len(same)):
        if type(same[i]) != np.int64:
            for j in range(len(same[i])):
                same2.append(same[i][j])
        else:
            same2.append(same[i])
    same2 = list(set(same2))
    return same2


def handle_data(data):
    x = copy.deepcopy(data)
    data_merge = pd.get_dummies(x.TOOL_ID)
    data_merge1 = pd.get_dummies(x['Tool'], prefix='Tool')
    data_merge2 = pd.get_dummies(x['TOOL_ID (#1)'], prefix='TOOL_1')
    data_merge3 = pd.get_dummies(x['TOOL_ID (#2)'], prefix='TOOL_2')
    data_merge4 = pd.get_dummies(x['TOOL_ID (#3)'], prefix='TOOL_3')
    data_merge5 = pd.get_dummies(x['Tool (#2)'], prefix='Tool_2')
    data_merge6 = pd.get_dummies(x['tool (#1)'], prefix='tool_1')
    data_merge7 = pd.get_dummies(x['TOOL'], prefix='TOOL')
    data_merge8 = pd.get_dummies(x['TOOL (#1)'], prefix='TOOL_#1')
    data_merge9 = pd.get_dummies(x['TOOL (#2)'], prefix='TOOL_#2')
    del x['TOOL_ID'], x['Tool'], x['TOOL_ID (#1)'], x['TOOL_ID (#2)'], x['TOOL_ID (#3)'], x['Tool (#2)'], x['tool (#1)'], x['TOOL'], x['TOOL (#1)'], x['TOOL (#2)']
    data_output = pd.concat([x, data_merge, data_merge1, data_merge2, data_merge3, data_merge4, data_merge5, data_merge6, data_merge7, data_merge8, data_merge9], axis=1)
    return data_output


def calc_turkey_test(x, y, z):
    data = copy.deepcopy(x)
    feature_list = copy.deepcopy(y)
    turkey_test_para = copy.deepcopy(z)
    turkey_test = pd.DataFrame(feature_list).set_index(0)
    turkey_test['turkey_test_min'] = np.nan
    turkey_test['turkey_test_max'] = np.nan
    turkey_test['turkey_test_min_num'] = 0
    turkey_test['turkey_test_max_num'] = 0
    start = time.time()
    for each_feature in feature_list:
        #print (each_feature)
        feature_data = data[each_feature]
        Q1 = feature_data.quantile(q=0.25, interpolation='linear')
        Q3 = feature_data.quantile(q=0.75, interpolation='linear')
        turkey_test_min = Q1 - turkey_test_para * (Q3 - Q1)
        turkey_test_max = Q3 + turkey_test_para * (Q3 - Q1)
        min_num = sum(feature_data < turkey_test_min)
        max_num = sum(feature_data > turkey_test_max)

        turkey_test.loc[each_feature, 'turkey_test_min'] = turkey_test_min
        turkey_test.loc[each_feature, 'turkey_test_max'] = turkey_test_max
        turkey_test.loc[each_feature, 'turkey_test_min_num'] = min_num
        turkey_test.loc[each_feature, 'turkey_test_max_num'] = max_num
    end = time.time()
    print('turkey test time cost:%s' % (end - start))
    return turkey_test


def calc_same_(x, y):
    feature_drop = copy.deepcopy(x)
    columns_dictionary = copy.deepcopy(y)
    feature_drop = pd.DataFrame(feature_drop)
    feature_groupby = feature_drop.groupby(0)
    feature_same = defaultdict(list)
    for first, second in feature_groupby:
        if first in feature_same.keys():
            feature_same[first].append(second[1].values)
        else:
            feature_same[first] = second[1].values
    feature_drop = defaultdict(list)
    same = []
    for i in feature_same:
        if i in same:
            continue
        for j in feature_same[i]:
            if j in feature_same.keys():
                same = calc_append(same, feature_same[i])
                same = calc_append(same, feature_same[j])
                if i in feature_drop.keys():
                    feature_drop[i] = calc_append(feature_drop[i], feature_same[j])
                else:
                    feature_drop[i] = list(feature_same[i])
                    feature_drop[i] = calc_append(feature_drop[i], feature_same[j])
    drop = []
    for i in feature_drop.keys():
        for j in feature_drop[i]:
            drop.append(j)
    drop = [columns_dictionary.get(item, item) for item in drop]
    return drop

def make_rf_predict(x, y, z, m):
    train = copy.deepcopy(x)
    test = copy.deepcopy(y)
    drop = copy.deepcopy(z)
    featurelist = copy.deepcopy(m)
    featurelist.drop(drop)
    ID = pd.DataFrame(test.ID).reset_index(drop=True)
    label = train.Y.values[:, np.newaxis]
#    len_d = len(train)
#    combine = np.concatenate((train[featurelist].values, test[featurelist].values), axis=0)
#    combine = minmax_scale(combine, axis=0)
#    train = combine[:len_d]
#    test = combine[len_d:]
    RF = RandomForestRegressor(n_estimators = 5000, oob_score=True, n_jobs=-1, verbose=1, max_features='sqrt')
    RF.fit(train[feature_list], label)
    model_output = RF.predict(test[feature_list])
    output = pd.concat([ID, pd.DataFrame(model_output).reset_index(drop=True)], axis=1)
    return output
    
data_origin = pd.read_excel('训练.xlsx')
data = handle_data(data_origin)
data_testA_origin = pd.read_excel('测试A.xlsx')
data_testA = handle_data(data_testA_origin)
data_testB_origin = pd.read_excel('测试A.xlsx')
data_testB = handle_data(data_testB_origin)
offline_train_origin = pd.read_csv('offline_train.csv')
offline_train = handle_data(offline_train_origin)
offline_test_origin = pd.read_csv('offline_test.csv')
offline_test = handle_data(offline_test_origin)
for i in [data_testA, data_testB, offline_train, offline_test]:
    for j in data.columns:
        if j not in i.columns:
            i[j] = 0
data_testA = data_testA[data.columns]
data_testB = data_testB[data.columns]
offline_train = offline_train[data.columns]
offline_test = offline_test[data.columns]

label = data.Y
ID = data.ID

feature_list = data.columns.drop('Y').drop('ID')
drop_same_feature = []
for each_feature in feature_list:
    #print (each_feature)
    if ~data[each_feature].isnull().values.any():
        if data[each_feature].min() == data[each_feature].max() or str(data[each_feature].min()) == 'nan':
            drop_same_feature.append(each_feature)

feature_list = feature_list.drop(drop_same_feature)

# data = data.replace(0, np.nan)  #replace 0 with NAN

#feature_list = pd.DataFrame(feature_list)

# for each_feature in ['TOOL_ID (#2)']:
#    feature_data = data[each_feature]
#
#    plt.figure()
#    plt.bar(feature_data.values, np.linspace(feature_data.min(), feature_data.max(), 500))
#    plt.show()
start = time.time()
feature_drop, columns_dictionary = calc_same(feature_list, data[feature_list])
drop = calc_same_(feature_drop, columns_dictionary)
feature_list = feature_list.drop(drop)
end = time.time()
print(end - start)

#%%  person
feature_person = []
for i in feature_list:
    person = sc.stats.pearsonr(data[i],label)
    feature_person.append([i, person[0]])
feature_person = pd.DataFrame(feature_person)
feature_drop = []
for i in feature_list:
    if data[i].isnull().sum().sum() != 0 or data_testA[i].isnull().sum().sum() != 0 or data_testB[i].isnull().sum().sum() != 0:
        feature_drop.append(i)
#feature_list = feature_list.drop(feature_drop)
feature_drop2 = []
#for i in feature_person.iterrows():
#    if abs(i[1][1]) < 0.01 :
#        feature_drop2.append(i[1][0])
feature_drop = list(set(feature_drop) | set(feature_drop2))
feature_list = feature_list.drop(feature_drop)
#%%
#data_scale = minmax_scale(data[feature_list])
#data_scale = pd.DataFrame(data_scale)
#data_scale.columns = feature_list
std = []
for i in feature_list:
    std.append([i, data[i].std()])
std = pd.DataFrame(std)
drop = list(std[std[1] < 1][0])
feature_list = feature_list.drop(drop)
#%%  Tukey's test
'''
turkey_test_para = 3
turkey_test = calc_turkey_test(data, feature_list, turkey_test_para)
'''
#%%
'''
group_avg = defaultdict(lambda x: 0)
turkey_test_para = 3
for i in data_origin.TOOL_ID.drop_duplicates():
    group_avg[i] = calc_turkey_test(data[data[i] == 1], feature_list, turkey_test_para)
group_drop = defaultdict(lambda x: 0)
for i in group_avg:
    feature_list_backup = copy.deepcopy(feature_list)
    each_data = data[data_origin.TOOL_ID == i]
    feature_drop, columns_dictionary = calc_same(feature_list_backup, each_data[feature_list_backup])
    drop = calc_same_(feature_drop, columns_dictionary)
    group_drop[i] = feature_list_backup.drop(drop)
'''
#%%  offline
offline_train = offline_train.fillna(0)
offline_test = offline_test.fillna(0)

#RF = RandomForestRegressor(n_estimators = 5000, oob_score=True, n_jobs=-1, verbose=1)
#RF.fit(offline_train[feature_list], offline_train.Y)
#test_predict = RF.predict(offline_test[feature_list])
#score = mean_squared_error(offline_test.Y.values, test_predict)
#print ('score:%s' % score)
'''
RF = RandomForestRegressor(n_jobs=-1)
params = {'n_estimators': [500,2000,4000], 'max_depth': [2, 5, 10, 20, None],
          'min_samples_split': [2, 5, 10, 15], 'min_samples_leaf': [2, 8, 15, 25],
          'max_features': ['auto', 'sqrt', 'log2', None]}
cv = GridSearchCV(RF, params, scoring = 'neg_mean_squared_error', verbose=3, oob_score=True, cv=ShuffleSplit(n_splits=5, test_size=0.2))
cv.fit(data[feature_list], data.Y)
score = cv.cv_results_
score.to_csv('score.csv', index=None)
'''
#xgb_train = xgb.DMatrix(offline_train[feature_list], label=offline_train.Y)
#xgb_test = xgb.DMatrix(offline_test[feature_list])
#params = {
#    'objective': 'reg:linear',
#    'eval_metric': 'rmse',
#    #'gamma': 6,
#    'slient': 1,
#    #'max_depth': 25,
#    'eta': 0.05,
#    'nthread': -1}
#watchlist = [(xgb_train, 'train'), (xgb_test, 'offline_test')]
#xgb_model_offline = xgb.train(params, xgb_train, num_boost_round=5000, early_stopping_rounds=50, evals=watchlist)
#xgb_output = xgb_model_offline.predict(xgb_test)
#score = mean_squared_error(offline_test.Y.values, xgb_output)
#print('score:%s' % score)

#%%
'''
data = data.fillna(0)
data_testA = data_testA.fillna(0)
#data = pd.concat([pd.DataFrame(ID), pd.DataFrame(data[feature_list]), pd.DataFrame(label)], axis=1)
xgb_train = xgb.DMatrix(data[feature_list], label=data.Y)
xgb_predict = xgb.DMatrix(data_testA[feature_list])
params = {
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    #'gamma': 6,
    'slient': 1,
    #'max_depth': 25,
    'eta': 0.05,
    'nthread': -1}
xgb_model = xgb.train(params, xgb_train, num_boost_round=5000)
xgb_output = xgb_model.predict(xgb_predict)
output = pd.concat([pd.DataFrame(data_testA_origin.ID), pd.DataFrame(xgb_output)], axis=1)
name = 'predict-' + str(time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))) + '.csv'
#output.to_csv(name, index=None, header=None)
'''
'''
predict_classify = pd.DataFrame(np.zeros((0,0),dtype=int))
for i in group_drop:
    data_train = data[data_origin.TOOL_ID == i]
    data_test = data_testA[data_testA_origin.TOOL_ID == i]
    if len(data_test) == 0:
        continue
    predict_classify_2 = make_rf_predict(data_train, data_test, group_drop[i], feature_list)
    predict_classify = pd.concat([predict_classify, predict_classify_2], axis=0)
'''
LR_train = minmax_scale(data[feature_list].values)
LR_pred = minmax_scale(data_testA[feature_list].values)
LR = Ridge(normalize=True, alpha=5.0)
LR.fit(data[feature_list], label)
predict_LR = LR.predict(data_testA[feature_list])
predict_LR = pd.concat([pd.DataFrame(data_testA_origin.ID), pd.DataFrame(predict_LR)], axis=1)
name = 'predict-LR-' + str(time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))) + '.csv'
predict_LR.to_csv(name, index=None, header=None)
RF = RandomForestRegressor(n_estimators=500, oob_score=True, n_jobs=-1, verbose=1)
RF.fit(data[feature_list], label)
predict_RF = RF.predict(data_testA[feature_list])
predict_RF = pd.concat([pd.DataFrame(data_testA_origin.ID), pd.DataFrame(predict_RF)], axis=1)
name = 'predict-RF-' + str(time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))) + '.csv'
predict_RF.to_csv(name, index=None, header=None)
GBDT = GradientBoostingRegressor(learning_rate=0.01, n_estimators=500, criterion ='mse', verbose=1)
GBDT.fit(data[feature_list], label)
predict_GBDT = GBDT.predict(data_testA[feature_list])
predict_GBDT = pd.concat([pd.DataFrame(data_testA_origin.ID), pd.DataFrame(predict_GBDT)], axis=1)
name = 'predict-GBDT-' + str(time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))) + '.csv'
predict_GBDT.to_csv(name, index=None, header=None)

LR_offline = LR.predict(data[feature_list])
LR_score = mean_squared_error(label, LR_offline)
RF_offline = RF.predict(data[feature_list])
RF_score = mean_squared_error(label, RF_offline)
GBDT_offline = GBDT.predict(data[feature_list])
GBDT_score = mean_squared_error(label, GBDT_offline)
print ('LR score:', LR_score)
print ('RF score:', RF_score)
print ('GBDT score:', GBDT_score)

importance_RF = RF.feature_importances_
