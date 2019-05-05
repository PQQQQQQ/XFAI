# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 13:57:02 2018

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

data = pd.read_csv('base_feature_z.csv')

data['area'] = data['creative_height'] * data['creative_width']
# data['os_osv'] = data['os'].astype(str).values + '_' + data['osv'].astype(str).values    

ad_cate_feature = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_first', 'advert_industry_inner_second', 
                   'advert_name', 'campaign_id', 'creative_id', 'creative_type', 'creative_tp_dnf', 'creative_has_deeplink',
                   'creative_is_jump', 'creative_is_download', 'area', 'os_osv']

media_cate_feature = ['app_cate_id', 'f_channel', 'app_id', 'inner_slot_id']

content_cate_feature = ['city', 'carrier', 'province', 'nnt', 'devtype', 'osv', 'os', 'make', 'model']

origin_cate_list = ad_cate_feature + media_cate_feature + content_cate_feature    


# 生成feature_click_count特征

# for feat in origin_cate_list:
#     data_c = data.groupby(feat)['click'].agg({'count'}).reset_index()
#     data_c[feat+'_count'] = data_c['count']
#     data_c = data_c.drop('count', axis=1)
#     data = pd.merge(data, data_c, how='left', on=[feat])

# data.to_csv('feature_click_count_os.csv', index=False)

# data_click = pd.read_csv('feature_click_count_os.csv')


# 生成feature_count特征

# for feat in origin_cate_list:
#     data_n = data_click[feat].value_counts().reset_index()
#     data_n.columns = [feat, feat+'_n_count']
#     data_click = pd.merge(data_click, data_n, how='left', on=[feat])
    
# data_click.to_csv('feature_count_os.csv', index=False)

for feat in origin_cate_list:
    data_n = data[feat].value_counts().reset_index()
    data_n.columns = [feat, feat+'_n_count']
    data = pd.merge(data, data_n, how='left', on=[feat])

data.to_csv('feature_count_os_z.csv', index=False)