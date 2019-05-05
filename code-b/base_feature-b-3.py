# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 17:17:34 2018

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

data = data.drop_duplicates().reset_index()
data = data.drop('index', axis=1)

data['day'] = data['time'].apply(lambda x: int(time.strftime("%d", time.localtime(x))))
data['hour'] = data['time'].apply(lambda x: int(time.strftime("%H", time.localtime(x))))

data['advert_industry_inner_first'] = data['advert_industry_inner'].apply(lambda x:x.split('_')[0]).apply(int)
data['advert_industry_inner_second'] = data['advert_industry_inner'].apply(lambda x:x.split('_')[1]).apply(int)

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
data['wh_ratio'] = data['creative_width'] / data['creative_height']
data['os_osv'] = data['os'].astype(str).values + '_' + data['osv'].astype(str).values


# 第二批新加的特征

data['inner_slot_id_first'] = data['inner_slot_id'].apply(lambda x:x.split('_')[0]) 
data['adid_1'] = data['adid']/10000
data['adid_1'] = data['adid_1'].astype(int)
data['orderid_1'] = data['orderid']/10000
data['orderid_1'] = data['orderid_1'].astype(int)
data['creative_id_1'] = data['creative_id']/10000
data['creative_id_1'] = data['creative_id_1'].astype(int)
data['app_id_1'] = data['app_id']/10000
data['app_id_1'] = data['app_id_1'].astype(int)



data['carrier_nnt'] = data['carrier'].astype(str).values + '_' + data['nnt'].astype(str).values #运营商和网络状态拼接

# 新加特征 这几个特征也可以尝试做count和unique
data['appid_adid'] = data['app_id'] + data['adid'] #不同app和不同ad的组合
data['creativeid_adid'] = data['creative_id'] + data['adid'] #不同创意id和不同ad的组合
data['orderid_adid'] = data['orderid'] + data['adid'] #不同订单id和不同ad的组合
data['advertid_adid'] = data['advert_id']+ data['adid']  #广告id和广告主id的组合
data['campaignid_adid'] = data['campaign_id'] + data['adid'] #活动id和广告id的组合

data['make_new'] = data['make'].str.split(',',expand = True)[0].str.split(' ',expand=True)[0].str.upper()

data['ad'] = data['adid'] + data['advert_industry_inner_first']


ad_feature = ['adid','advert_id','orderid','advert_industry_inner',
           'campaign_id','creative_id','creative_type','creative_tp_dnf',
           'creative_is_jump','creative_is_download','creative_has_deeplink','advert_name']

media_feature = ['app_id','app_cate_id','inner_slot_id','f_channel']

user_feature = ['city','province','carrier','make','model','nnt','os','os_name','osv','devtype']

# adid和用户、媒体特征的组合特征
id_nuq = user_feature + media_feature


# 组合count特征
for feature in id_nuq:
    feature_name = 'ad_' + feature + '_count'  
    temp = data.groupby(['ad',feature]).size().reset_index().rename(columns={0: feature_name})
    data = pd.merge(data, temp, how='left', on=['ad',feature])

for feature in user_feature:
    feature_name = 'adid_' + feature + '_count'  
    temp = data.groupby(['adid',feature]).size().reset_index().rename(columns={0: feature_name})
    data = pd.merge(data, temp, how='left', on=['adid',feature])

for feature in user_feature:
    feature_name = 'appid_' + feature + '_count'  
    temp = data.groupby(['app_id',feature]).size().reset_index().rename(columns={0: feature_name})
    data = pd.merge(data, temp, how='left', on=['app_id',feature])

for feature in user_feature:
    feature_name = 'orderid_' + feature + '_count'  
    temp = data.groupby(['orderid',feature]).size().reset_index().rename(columns={0: feature_name})
    data = pd.merge(data, temp, how='left', on=['orderid',feature])

for feature in user_feature:
    feature_name = 'creativeid_' + feature + '_count'  
    temp = data.groupby(['creative_id',feature]).size().reset_index().rename(columns={0: feature_name})
    data = pd.merge(data, temp, how='left', on=['creative_id',feature])

for feature in user_feature:
    feature_name = 'advertid_' + feature + '_count'  
    temp = data.groupby(['advert_id',feature]).size().reset_index().rename(columns={0: feature_name})
    data = pd.merge(data, temp, how='left', on=['advert_id',feature]) 

for feature in user_feature:
    feature_name = 'campaignid_' + feature + '_count'  
    temp = data.groupby(['campaign_id',feature]).size().reset_index().rename(columns={0: feature_name})
    data = pd.merge(data, temp, how='left', on=['campaign_id',feature]) 


# unique特征  ad
for feature in id_nuq:
    feature_name_1 = 'ad_' + feature +'_unique'
    feature_name_2 = feature + '_ad' + '_unique'
    gp1=data.groupby('ad')[feature].nunique().reset_index().rename(columns={feature:feature_name_1})
    gp2=data.groupby(feature)['ad'].nunique().reset_index().rename(columns={'ad':feature_name_2})
    data=pd.merge(data,gp1,how='left',on=['ad'])
    data=pd.merge(data,gp2,how='left',on=[feature])  


'''=================================================================================================================='''



ad_cate_feature = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_first', 'advert_industry_inner_second', 
                   'advert_name', 'campaign_id', 'creative_id', 'creative_type', 'creative_tp_dnf', 'creative_has_deeplink',
                   'creative_is_jump', 'creative_is_download', 'area', 'os_osv', 'wh_ratio']

media_cate_feature = ['app_cate_id', 'f_channel', 'app_id', 'inner_slot_id', 'inner_slot_id_first']

content_cate_feature = ['city', 'carrier', 'province', 'nnt', 'devtype', 'osv', 'os', 'os_name', 'make', 'model']

add_feature = ['adid_1', 'orderid_1', 'creative_id_1', 'app_id_1']

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
data1.to_csv('feature_count_b3.csv', index=False)
data = pd.read_csv('feature_count_b3.csv')


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

data.to_csv('base_feature-b_3.csv', index=False)