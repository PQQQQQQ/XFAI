# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 13:49:19 2018

@author: PQ
"""

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from scipy import sparse
import lightgbm as lgb
import numpy as np
import pandas as pd
import time
import datetime
import gc
import os

train = pd.read_table('round1_iflyad_train.txt')
predict = pd.read_table('round1_iflyad_test_feature.txt')

# train['time'] = train['time'].apply(lambda x : time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(x)))
# train['time'] = pd.to_datetime(train['time'])

# predict['time'] = predict['time'].apply(lambda x : time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(x)))
# predict['time'] = pd.to_datetime(predict['time'])

predict['click']=-1

data = pd.concat([train,predict], axis=0, ignore_index=True)
# data["day"]=data["time"].dt.day
# data["hour"]=data["time"].dt.hour

data['day'] = data['time'].apply(lambda x: int(time.strftime("%d", time.localtime(x))))
data['hour'] = data['time'].apply(lambda x: int(time.strftime("%H", time.localtime(x))))

# data['inner_slot_id_first'] = data['inner_slot_id'].apply(lambda x:x.split('_')[0])
# data['inner_slot_id_second'] = data['inner_slot_id'].apply(lambda x:x.split('_')[1])

data['advert_industry_inner_first'] = data['advert_industry_inner'].apply(lambda x:x.split('_')[0])
data['advert_industry_inner_second'] = data['advert_industry_inner'].apply(lambda x:x.split('_')[1])

# replace
replace = ['creative_is_jump', 'creative_is_download', 'creative_is_js', 'creative_is_voicead', 'creative_has_deeplink', 'app_paid']
for feat in replace:
    data[feat] = data[feat].replace([False, True], [0, 1])
    
# fillna
data['make'] = data['make'].fillna(data['make'].mode()[0])
data['model'] = data['model'].fillna(data['model'].mode()[0])
data['f_channel'] = data['f_channel'].fillna(data['f_channel'].mode()[0])
# data['f_channel'] = data['f_channel'].fillna(str(-1))
data['osv'] = data['osv'].fillna(data['osv'].mode()[0])
data['app_id'] = data['app_id'].fillna(data['app_id'].mode()[0])
data['app_cate_id'] = data['app_cate_id'].fillna(data['app_cate_id'].mean())
data['user_tags'] = data['user_tags'].fillna(str(-1))

# data['make'] = data['make'].fillna(str(-1))
# data['model'] = data['model'].fillna(str(-1))
# data['f_channel'] = data['f_channel'].fillna(str(-1))
# data['osv'] = data['osv'].fillna(str(-1))
# data['app_id'] = data['app_id'].fillna(str(-1))
# data['app_cate_id'] = data['app_cate_id'].fillna(str(-1))
# data['user_tags'] = data['user_tags'].fillna(str(-1))

data = data.drop(['creative_is_js','creative_is_voicead','app_paid'], axis=1)
data['area'] = data['creative_height'] * data['creative_width']
data = data.drop('advert_industry_inner', axis=1)
data['os_osv'] = data['os'].astype(str).values + '_' + data['osv'].astype(str).values

# 去掉os_osv多余的噪声
# 去掉os, osv运行

# temp = data['os_osv'].value_counts().reset_index()
# temp_set = set(temp[temp['os_osv'] == 1]['index'].values)
# data['os_osv'] = data['os_osv'].map(lambda x: '1_11_2' if x in temp_set else x)

# temp1 = data['make'].value_counts().reset_index()
# temp1_set = set(temp1[temp1['make'] == 1]['index'].values)
# data['make'] = data['make'].map(lambda x: 'XDL' if x in temp_set else x)

# temp2 = data['model'].value_counts().reset_index()
# temp2_set = set(temp2[temp2['model'] == 1]['index'].values)
# data['model'] = data['model'].map(lambda x: 'V9C' if x in temp_set else x)


ad_cate_feature = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_first', 'advert_industry_inner_second', 
                   'advert_name', 'campaign_id', 'creative_id', 'creative_type', 'creative_tp_dnf', 'creative_has_deeplink',
                   'creative_is_jump', 'creative_is_download', 'area', 'os_osv']

media_cate_feature = ['app_cate_id', 'f_channel', 'app_id', 'inner_slot_id']

content_cate_feature = ['city', 'carrier', 'province', 'nnt', 'devtype', 'osv', 'os', 'make', 'model']

origin_cate_list = ad_cate_feature + media_cate_feature + content_cate_feature

for i in origin_cate_list:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(0, data[i].nunique()))))


data = data.drop('os_name',axis=1)
data['period'] = data['day']
data['period'][data['period']<27] = data['period'][data['period']<27] + 31
data = data.drop('time', axis=1)
data = data.drop('day', axis=1)

# one_hot_feature = ['city', 'province', 'make', 'model', 'osv', 'adid', 'advert_id', 'orderid',
#            'advert_industry_inner_first', 'advert_industry_inner_second', 'campaign_id', 'creative_id', 'app_cate_id',
#            'app_id', 'inner_slot_id', 'advert_name', 'f_channel', 'creative_tp_dnf', 'user_tags']

# for feature in one_hot_feature:
#     try:
#         data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
#     except:
#         data[feature] = LabelEncoder().fit_transform(data[feature])
        

# 构造历史点击率特征
for feat_1 in origin_cate_list:
# for feat_1 in ['advert_id','advert_industry_inner_first', 'advert_industry_inner_second', 'advert_name', 'campaign_id', 'creative_height',
#                'creative_tp_dnf', 'creative_width', 'province', 'f_channel']:
    gc.collect()
    res = pd.DataFrame()
    temp = data[[feat_1, 'period', 'click']]
    for period in range(27, 35):
        if period == 27:
            count=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<=period).values].count()).reset_index(name=feat_1+'_all')
            count1=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<=period).values].sum()).reset_index(name=feat_1+'_1')
        else: 
            count=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<period).values].count()).reset_index(name=feat_1+'_all')
            count1=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<period).values].sum()).reset_index(name=feat_1+'_1')
        count[feat_1+'_1']=count1[feat_1+'_1']
        count.fillna(value=0, inplace=True)
        count[feat_1+'_rate'] = round(count[feat_1+'_1'] / count[feat_1+'_all'], 5)
        count['period'] = period
        count.drop([feat_1+'_all', feat_1+'_1'], axis=1, inplace=True)
        count.fillna(value=0, inplace=True)
        res=res.append(count, ignore_index=True)
    print(feat_1, 'over')
    data = pd.merge(data, res, how='left', on=[feat_1, 'period'])

data.to_csv('base_feature_z.csv', index=False)


# base_feature_best.py
'''--------------------------------------------------------------------------------------------'''