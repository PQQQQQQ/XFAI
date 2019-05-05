# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 15:30:47 2018

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

train1 = pd.read_table('round2_iflyad_train.txt')
train2 = pd.read_table('round1_iflyad_train.txt')
predict = pd.read_table('round2_iflyad_test_feature.txt')

predict['click']=-1

data = pd.concat([train1, train2, predict], axis=0, ignore_index=True)


data['day'] = data['time'].apply(lambda x: int(time.strftime("%d", time.localtime(x))))
data['hour'] = data['time'].apply(lambda x: int(time.strftime("%H", time.localtime(x))))

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


data = data.drop(['creative_is_js','creative_is_voicead','app_paid'], axis=1)
data['area'] = data['creative_height'] * data['creative_width']
data = data.drop('advert_industry_inner', axis=1)
data['os_osv'] = data['os'].astype(str).values + '_' + data['osv'].astype(str).values



ad_cate_feature = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_first', 'advert_industry_inner_second', 
                   'advert_name', 'campaign_id', 'creative_id', 'creative_type', 'creative_tp_dnf', 'creative_has_deeplink',
                   'creative_is_jump', 'creative_is_download', 'area', 'os_osv']

media_cate_feature = ['app_cate_id', 'f_channel', 'app_id', 'inner_slot_id']

content_cate_feature = ['city', 'carrier', 'province', 'nnt', 'devtype', 'osv', 'os', 'os_name', 'make', 'model']

origin_cate_list = ad_cate_feature + media_cate_feature + content_cate_feature

for i in origin_cate_list:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(0, data[i].nunique()))))



data['period'] = data['day']
data['period'][data['period']<27] = data['period'][data['period']<27] + 31
data = data.drop('time', axis=1)
data = data.drop('day', axis=1)




# 构造时间点击率特征

data1 = data.copy()
data1['click'] = data1['click'].replace(-1, 0)
for feat in origin_cate_list:
# for feat_1 in ['advert_id','advert_industry_inner_first', 'advert_industry_inner_second', 'advert_name', 'campaign_id', 'creative_height',
#                'creative_tp_dnf', 'creative_width', 'province', 'f_channel']:
    gc.collect()
    res = pd.DataFrame()
    temp = data1[[feat, 'hour', 'click']]
    for hour in range(0, 24):
        if hour == 0:
            count=temp.groupby([feat]).apply(lambda x: x['click'][(x['hour']<=hour).values].count()).reset_index(name=feat+'_h_all')
            count1=temp.groupby([feat]).apply(lambda x: x['click'][(x['hour']<=hour).values].sum()).reset_index(name=feat+'_h_1')
        else: 
            count=temp.groupby([feat]).apply(lambda x: x['click'][(x['hour']<hour).values].count()).reset_index(name=feat+'_h_all')
            count1=temp.groupby([feat]).apply(lambda x: x['click'][(x['hour']<hour).values].sum()).reset_index(name=feat+'_h_1')
        count[feat+'_h_1']=count1[feat+'_h_1']
        count.fillna(value=0, inplace=True)
        count[feat+'_h_rate'] = round(count[feat+'_h_1'] / count[feat+'_h_all'], 5)
        count['hour'] = hour
        count.drop([feat+'_h_all', feat+'_h_1'], axis=1, inplace=True)
        count.fillna(value=0, inplace=True)
        res=res.append(count, ignore_index=True)
    print(feat+'_h', 'over')
    data1 = pd.merge(data1, res, how='left', on=[feat, 'hour'])

data1['click'] = data['click']
data1.to_csv('feature_count_b.csv', index=False)
data = pd.read_csv('feature_count_b.csv')


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

data.to_csv('base_feature-b_h.csv', index=False)
# data = pd.read_csv('feature_count_d.csv')

# data1 = data1.drop('click', axis=1)

# data_h = pd.merge(data, data1, how='left',on=['instance_id', 'city', 'province', 'user_tags', 'carrier', 'devtype',
#        'make', 'model', 'nnt', 'os', 'osv', 'adid', 'advert_id', 'orderid', 'campaign_id', 'creative_id', 'creative_tp_dnf', 'app_cate_id',
#        'f_channel', 'app_id', 'inner_slot_id', 'creative_type', 'creative_width', 'creative_height', 'creative_is_jump',
#        'creative_is_download', 'creative_has_deeplink', 'advert_name', 'hour', 'advert_industry_inner_first', 'advert_industry_inner_second',
#        'area', 'os_osv', 'period'])

# data_h.to_csv('base_feature-b_h.csv', index=False)