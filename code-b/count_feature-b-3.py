# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 19:53:23 2018

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


data = pd.read_csv('base_feature-b_3.csv')

data['area'] = data['creative_height'] * data['creative_width']
# data['wh_ratio'] = data['creative_width'] / data['creative_height']
# data['os_osv'] = data['os'].astype(str).values + '_' + data['osv'].astype(str).values    

ad_cate_feature = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_first', 'advert_industry_inner_second', 
                   'advert_name', 'campaign_id', 'creative_id', 'creative_type', 'creative_tp_dnf', 'creative_has_deeplink',
                   'creative_is_jump', 'creative_is_download', 'area', 'os_osv', 'wh_ratio']

media_cate_feature = ['app_cate_id', 'f_channel', 'app_id', 'inner_slot_id', 'inner_slot_id_first']

content_cate_feature = ['city', 'carrier', 'province', 'nnt', 'devtype', 'osv', 'os', 'os_name', 'make', 'model']

# add_feature = ['adid_1', 'orderid_1', 'creative_id_1', 'app_id_1']

origin_cate_list = ad_cate_feature + media_cate_feature + content_cate_feature  

# id_feature = ['adid', 'advert_id', 'orderid', 'campaign_id', 'creative_id', 'creative_tp_dnf', 'app_cate_id', 'app_id', 'inner_slot_id']

# cnt_feature = ['adid', 'city', 'carrier', 'province', 'nnt', 'devtype', 'osv', 'os', 'make', 'model'] 

# col_type = cnt_feature.copy()
# n = len(col_type)

# 生成feature_click_count特征

# for feat in id_feature:
#     data_c = data.groupby(feat)['click'].agg({'count'}).reset_index()
#     data_c[feat+'_count'] = data_c['count']
#     data_c = data_c.drop('count', axis=1)
#     data = pd.merge(data, data_c, how='left', on=[feat])

# for i in range(n):
#     for j in range(n):
#         if i!= j:
#             data_c = data.groupby([col_type[i], col_type[j]]).size().reset_index()
#             data_c = data_c.rename(columns={0:col_type[i]+'_in_'+col_type[j]+'_count'})
#             data = pd.merge(data, data_c, how='left', on=[col_type[i], col_type[j]])

# data.to_csv('feature_in_count_1015.csv', index=False)

# data_count = pd.read_csv('feature_in_count_1015.csv')


# 生成feature_count特征

for feat in origin_cate_list:
    data_n = data[feat].value_counts().reset_index()
    data_n.columns = [feat, feat+'_n_count']
    data = pd.merge(data, data_n, how='left', on=[feat])
    
data.to_csv('feature_count_1016.csv', index=False)