# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 10:06:25 2018

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
import warnings
import numpy as np
import pandas as pd
import time
import datetime
import gc
import os


warnings.filterwarnings("ignore")


data = pd.read_csv('feature_count_1016.csv')

train = data[data.click != -1]

ad_cate_feature = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_first', 'advert_industry_inner_second', 
                   'advert_name', 'campaign_id', 'creative_id', 'creative_type', 'creative_tp_dnf', 'creative_has_deeplink',
                   'creative_is_jump', 'creative_is_download', 'area', 'os_osv']

media_cate_feature = ['app_cate_id', 'f_channel', 'app_id', 'inner_slot_id']

content_cate_feature = ['city', 'carrier', 'province', 'nnt', 'devtype', 'os', 'osv', 'os_name', 'make', 'model']

# add_feature = ['adid_1', 'orderid_1', 'creative_id_1', 'app_id_1']
add_feature = ['ad', 'make_new']

for i in add_feature:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(0, data[i].nunique()))))

origin_cate_list = ad_cate_feature + media_cate_feature + content_cate_feature + add_feature



rate_feature = ['adid_rate', 'advert_id_rate', 'orderid_rate', 'advert_industry_inner_first_rate', 'advert_industry_inner_second_rate', 'advert_name_rate', 'campaign_id_rate', 
				'creative_id_rate', 'creative_type_rate', 'creative_tp_dnf_rate', 'creative_has_deeplink_rate','creative_is_jump_rate', 'creative_is_download_rate', 
				'app_cate_id_rate', 'f_channel_rate', 'app_id_rate', 'inner_slot_id_rate', 'city_rate', 'carrier_rate', 'province_rate', 'nnt_rate', 
                'devtype_rate', 'make_rate', 'model_rate', 'area_rate', 'os_rate', 'osv_rate', 'os_osv_rate', 'os_name_rate']


# rate_feature_h = ['adid_h_rate', 'advert_id_h_rate', 'orderid_h_rate', 'advert_industry_inner_first_h_rate', 'advert_industry_inner_second_h_rate', 'advert_name_h_rate', 
# 				          'campaign_id_h_rate', 'creative_id_h_rate', 'creative_type_h_rate', 'creative_tp_dnf_h_rate', 'creative_has_deeplink_h_rate', 'creative_is_jump_h_rate',
# 				          'creative_is_download_h_rate', 'area_h_rate', 'os_osv_h_rate', 'app_cate_id_h_rate', 'f_channel_h_rate', 'app_id_h_rate', 'inner_slot_id_h_rate', 'city_h_rate',
#                   'carrier_h_rate', 'province_h_rate', 'nnt_h_rate', 'devtype_h_rate', 'osv_h_rate', 'os_h_rate', 'os_name_h_rate', 'make_h_rate', 'model_h_rate']


id_count_feature = ['adid_n_count', 'advert_id_n_count', 'orderid_n_count', 'advert_industry_inner_first_n_count', 'advert_industry_inner_second_n_count', 
                    'advert_name_n_count', 'campaign_id_n_count', 'creative_id_n_count', 'creative_type_n_count', 'creative_tp_dnf_n_count', 'creative_has_deeplink_n_count',
                    'creative_is_jump_n_count', 'creative_is_download_n_count', 'app_cate_id_n_count', 'f_channel_n_count', 'app_id_n_count', 'inner_slot_id_n_count',
                    'city_n_count', 'carrier_n_count', 'province_n_count','nnt_n_count', 'devtype_n_count', 'make_n_count', 'model_n_count', 'area_n_count', 'os_osv_n_count',
                    'os_name_n_count']


combine_feature = ['ad_city_count', 'ad_province_count', 'ad_carrier_count',
       'ad_make_count', 'ad_model_count', 'ad_nnt_count', 'ad_os_count',
       'ad_os_name_count', 'ad_osv_count', 'ad_devtype_count',
       'ad_app_id_count', 'ad_app_cate_id_count',
       'ad_inner_slot_id_count', 'ad_f_channel_count', 
       'adid_city_count', 'adid_province_count', 'adid_carrier_count',
       'adid_make_count', 'adid_model_count', 'adid_nnt_count',
       'adid_os_count', 'adid_os_name_count', 'adid_osv_count',
       'adid_devtype_count', 'appid_city_count', 'appid_province_count',
       'appid_carrier_count', 'appid_make_count', 'appid_model_count',
       'appid_nnt_count', 'appid_os_count', 'appid_os_name_count',
       'appid_osv_count', 'appid_devtype_count', 'orderid_city_count',
       'orderid_province_count', 'orderid_carrier_count',
       'orderid_make_count', 'orderid_model_count', 'orderid_nnt_count',
       'orderid_os_count', 'orderid_os_name_count', 'orderid_osv_count',
       'orderid_devtype_count', 'creativeid_city_count',
       'creativeid_province_count', 'creativeid_carrier_count',
       'creativeid_make_count', 'creativeid_model_count',
       'creativeid_nnt_count', 'creativeid_os_count',
       'creativeid_os_name_count', 'creativeid_osv_count',
       'creativeid_devtype_count', 'advertid_city_count',
       'advertid_province_count', 'advertid_carrier_count',
       'advertid_make_count', 'advertid_model_count',
       'advertid_nnt_count', 'advertid_os_count',
       'advertid_os_name_count', 'advertid_osv_count',
       'advertid_devtype_count', 'campaignid_city_count',
       'campaignid_province_count', 'campaignid_carrier_count',
       'campaignid_make_count', 'campaignid_model_count',
       'campaignid_nnt_count', 'campaignid_os_count',
       'campaignid_os_name_count', 'campaignid_osv_count',
       'campaignid_devtype_count']


unique_feature = ['ad_city_unique', 'city_ad_unique',
       'ad_province_unique', 'province_ad_unique', 'ad_carrier_unique',
       'carrier_ad_unique', 'ad_make_unique', 'make_ad_unique',
       'ad_model_unique', 'model_ad_unique', 'ad_nnt_unique',
       'nnt_ad_unique', 'ad_os_unique', 'os_ad_unique',
       'ad_os_name_unique', 'os_name_ad_unique', 'ad_osv_unique',
       'osv_ad_unique', 'ad_devtype_unique', 'devtype_ad_unique',
       'ad_app_id_unique', 'app_id_ad_unique', 'ad_app_cate_id_unique',
       'app_cate_id_ad_unique', 'ad_inner_slot_id_unique',
       'inner_slot_id_ad_unique', 'ad_f_channel_unique',
       'f_channel_ad_unique'] 

# add_count_feature = ['adid_1_n_count', 'orderid_1_n_count', 'creative_id_1_n_count', 'app_id_1_n_count']

# click_count_feature = ['adid_count', 'advert_id_count', 'orderid_count', 'advert_industry_inner_first_count', 'advert_industry_inner_second_count', 
#                 'advert_name_count', 'campaign_id_count', 'creative_id_count', 'creative_type_count', 'creative_tp_dnf_count', 'creative_has_deeplink_count',
#                 'creative_is_jump_count', 'creative_is_download_count', 'app_cate_id_count', 'f_channel_count', 'app_id_count', 'inner_slot_id_count',
#                 'city_count', 'carrier_count', 'province_count','nnt_count', 'devtype_count', 'osv_count', 'os_count', 'make_count', 
#                 'model_count', 'area_count', 'os_osv_count']


# click_count_feature = ['adid_count', 'advert_id_count', 'orderid_count', 'campaign_id_count', 'creative_id_count', 'creative_tp_dnf_count', 'app_cate_id_count',
#                        'app_id_count', 'inner_slot_id_count']

# 点击次数特征对线上结果并没有什么提升



cate_feature = origin_cate_list + rate_feature

num_feature = ['creative_width', 'creative_height', 'hour', 'period']

num_feature = num_feature + id_count_feature + combine_feature + unique_feature

feature = cate_feature + num_feature
print(len(feature), feature)


predict = data[data.click == -1]
predict_result = predict[['instance_id']]
predict_result['predicted_score'] = 0
predict_x = predict.drop('click', axis=1)

train_x = data[data.click != -1]
train_y = data[data.click != -1].click.values

if os.path.exists('base_train_csr.npz') and False:
    print('load_csr---------')
    base_train_csr = sparse.load_npz('base_train_csr.npz').tocsr().astype('bool')
    base_predict_csr = sparse.load_npz('base_predict_csr.npz').tocsr().astype('bool')
else:
    base_train_csr = sparse.csr_matrix((len(train), 0))
    base_predict_csr = sparse.csr_matrix((len(predict_x), 0))

    enc = OneHotEncoder()
    for feature in cate_feature:
        enc.fit(data[feature].values.reshape(-1, 1))
        base_train_csr = sparse.hstack((base_train_csr, enc.transform(train_x[feature].values.reshape(-1, 1))), 'csr',
                                       'bool')
        base_predict_csr = sparse.hstack((base_predict_csr, enc.transform(predict[feature].values.reshape(-1, 1))),
                                         'csr',
                                         'bool')
    print('one-hot prepared !')
    
    cv = CountVectorizer(min_df=20)
    for feature in ['user_tags']:
        data[feature] = data[feature].astype(str)
        cv.fit(data[feature])
        base_train_csr = sparse.hstack((base_train_csr, cv.transform(train_x[feature].astype(str))), 'csr', 'bool')
        base_predict_csr = sparse.hstack((base_predict_csr, cv.transform(predict_x[feature].astype(str))), 'csr',
                                         'bool')
    print('cv prepared !')

    sparse.save_npz('base_train_csr.npz', base_train_csr)
    sparse.save_npz('base_predict_csr.npz', base_predict_csr)

train_csr = sparse.hstack(
    (sparse.csr_matrix(train_x[num_feature]), base_train_csr), 'csr').astype(
    'float32')
predict_csr = sparse.hstack(
    (sparse.csr_matrix(predict_x[num_feature]), base_predict_csr), 'csr').astype('float32')
print(train_csr.shape)
# feature_select = SelectPercentile(chi2, percentile=95)
# feature_select.fit(train_csr, train_y)
# train_csr = feature_select.transform(train_csr)
# predict_csr = feature_select.transform(predict_csr)
# print('feature select')
# print(train_csr.shape)



# 模型部分

lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=48, max_depth=-1, learning_rate=0.05, n_estimators=3000,
                           max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                           min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                           colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=10, silent=True)


# lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=48, max_depth=-1, reg_lambda=1, learning_rate=0.01,subsample_for_bin=50000,
#          n_estimators=3000, objective='binary', subsample=0.8, min_split_gain=0, subsample_freq=1,min_child_samples=10, colsample_bytree=1,
#          reg_alpha=0.0, min_child_weight=5, n_jobs=50, silent=True,  seed=1000, max_bin=425) 


skf = StratifiedKFold(n_splits=5, random_state=1024, shuffle=True)
best_score = []
loss = 0
for index, (train_index, test_index) in enumerate(skf.split(train_csr, train_y)):
    lgb_model.fit(train_csr[train_index], train_y[train_index],
                  eval_set=[(train_csr[train_index], train_y[train_index]),
                            (train_csr[test_index], train_y[test_index])], early_stopping_rounds=100)
    best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
    loss += lgb_model.best_score_['valid_1']['binary_logloss']
    print(best_score)
    test_pred = lgb_model.predict_proba(predict_csr, num_iteration=lgb_model.best_iteration_)[:, 1]
    print('test mean:', test_pred.mean())
    predict_result['predicted_score'] = predict_result['predicted_score'] + test_pred

print('logloss:', best_score, loss/5)

print(np.mean(best_score))
predict_result['predicted_score'] = predict_result['predicted_score'] / 5
mean = predict_result['predicted_score'].mean()
print('mean:', mean)
now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')
predict_result[['instance_id', 'predicted_score']].to_csv("XFAI-b-1%s.csv" % now, index=False)