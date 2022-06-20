



import pandas as pd
pd.options.display.max_columns = None
import numpy as np
import seaborn as sns
#sns.set(rc={'figure.figsize':(20,10)})
import matplotlib.pyplot as plt
import datetime 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, StandardScaler, MinMaxScaler
lbec = LabelEncoder()
scaler = StandardScaler()
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, plot_precision_recall_curve
from imblearn.over_sampling import SMOTE 
from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.under_sampling import (RandomUnderSampler, 
                                     ClusterCentroids,
                                     TomekLinks,
                                     NeighbourhoodCleaningRule,
                                     NearMiss)
import xgboost as xgb
from xgboost import plot_tree, XGBClassifier, plot_importance
import lightgbm as lgb
from lightgbm import plot_importance as lgbm_plt
#from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


import re
import pickle
import os
os.chdir(r"C:\Users\andyso\Desktop\Andy\python\prospect")


df = pd.read_csv('mx_chinese_prospect_list_v4.csv')
df.columns = ['member_account_id','spend_WEHK','spend_MNHK','spend_SEHK','spend_IKHK','spend_PHHK','spend_PHDHK','spend_KFCHK','spend_MXHK_nonC','txn_WEHK','txn_MNHK','txn_SEHK','txn_IKHK','txn_PHHK','txn_PHDHK','txn_KFCHK','txn_MXHK_nonC','apg_WEHK','apg_MNHK','apg_SEHK','apg_IKHK','apg_PHHK','apg_PHDHK','apg_KFCHK','apg_MXHK_nonC','txn_WEHK_impulse','spend_WEHK_impulse','txn_WEHK_produce','spend_WEHK_produce','txn_WEHK_ready','spend_WEHK_ready','txn_WEHK_staples','spend_WEHK_staples','txn_SEHK_beer','spend_SEHK_beer','txn_SEHK_dairy','spend_SEHK_dairy','txn_SEHK_frozen','spend_SEHK_frozen','txn_SEHK_instant','spend_SEHK_instant','txn_SEHK_media','spend_SEHK_media','MXHK_cuisine_cnt','MXHK_brand_cnt','spend_HASE_C','txn_HASE_C','is_prospect','linked_hase','linked_octopus','reporting_country','enroll_channel','enroll_state','on_boarding_days','RFM_segment','recency','frequency','monetary','CEM_score_gp','total_score','SOW_WEHK_impulse','SOW_WEHK_produce','SOW_WEHK_ready','SOW_WEHK_staples','SOW_SEHK_beer','SOW_SEHK_dairy','SOW_SEHK_frozen','SOW_SEHK_instant','SOW_SEHK_media']
#print(df.head())
print(df.columns)
print(df.shape)
print(df.info())
# print(df.describe().T)



def check_missing_col():
    #check missing values by column
    missing = df.isnull().sum()/ len(df) *100
    missing = missing[missing > 0]
    return(missing)

def remove_missing_col_data(df, missing_rate):
    missing = df.isnull().sum()/ len(df) *100
    missing = missing[missing > 0]
    missing80pct_cols = missing[missing > missing_rate].index.tolist()
    print('no of cols to drop: ' + str(len(missing80pct_cols)))
    return(df.drop(missing80pct_cols, axis=1))
## REMOVED
# apg_IKHK            92.232462
# apg_PHHK            96.075044
# apg_PHDHK           98.629272
# apg_KFCHK           87.950600
# apg_MXHK_nonC       82.630362

df = remove_missing_col_data(df,80)

missing = df.isnull().sum()/df.shape[0]
print(missing[missing > 0])


#drop single value column
to_drop = ['member_account_id']
for var in missing.index:
  if len(df[var].unique()) == 1:
    to_drop.append(var)

df = df.drop(to_drop, axis=1)
missing = missing.drop(labels=to_drop)

#check variable values
chk_df = {}
for var in missing.index:
  chk_df[var] = len(df[var].unique())

df = df.dropna(subset=['RFM_segment'])
# 3,304,671

df = df[df['on_boarding_days'] > 30]

chk_col = ['linked_hase', 'linked_octopus']
for var in chk_col:
    print(df[var].value_counts())
    

#fill sow missing value as 0
fill_col = [k for k in df.columns if ('SOW' in k) ]
df[fill_col] = df[fill_col].fillna(0)

#fill apg missing value as 9999
fill_col0 = [k for k in df.columns if ('apg' in k) ]
df[fill_col0] = df[fill_col0].fillna(9999)


#fill unknown to object variables
fill_col1 = ['enroll_state', 'enroll_channel']
for var in fill_col1:
  #print(df[var].mode)
  df[var] = df[var].fillna('unknown')

#fill mode
#df.groupby(['CEM_score_gp']).size()
#print(df['CEM_score_gp'].mode())

df['CEM_score_gp'] = df['CEM_score_gp'].fillna(df['CEM_score_gp'].mode())

#fill median
fill_col3 = ['recency', 'frequency', 'monetary', 'total_score']
for var in fill_col3:
  #print(df[var].median)
  df[var] = df[var].fillna(df[var].median())

# sns.distplot(df.SOW_WEHK_impulse)

#encoding
one_hot_var = ['reporting_country', 'enroll_channel', 'enroll_state']
label_var = ['RFM_segment', 'CEM_score_gp']

for var in label_var:    
  df['label_'+var] = lbec.fit_transform(df[var].astype(str))
df.drop(label_var, axis=1, inplace=True)

for var in one_hot_var:
  df = pd.get_dummies(df, columns=[var], prefix = ['OH_' + var])


"""final feature selection"""
def remove_correlated(df):
    corr_df = df.corr().abs()
    # Create and apply mask
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    tri_df = corr_df.mask(mask)
    tri_df.to_excel('df_correlation.xlsx')
    # Find columns that meet treshold
    to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.89)]
    print(to_drop)
    #df = df.drop(to_drop, axis=1)
    print("removed correlation >0.89")
    return(df.drop(to_drop, axis=1))
           
df = remove_correlated(df)
# ['txn_SEHK_instant', 'MXHK_cuisine_cnt', 'total_score', 'OH_reporting_country_Others', 'OH_enroll_channel_unknown']
# removed correlation >0.89

###########output cleansed data
df.to_csv('MX_chinese_prospect_cleansed.csv',index=False)

# df = pd.read_csv('MX_chinese_prospect_cleansed.csv')
# df = df.drop('Unnamed: 0',axis=1)

#modelling
# df.is_prospect.value_counts()
X = df.drop('is_prospect', axis=1)
y = df['is_prospect']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

"""lightGBM"""
clf = lgb.LGBMClassifier(objective=' y',
                          metric='logloss',
                          learning_rate = 0.05,
                          boosting_type= 'gbdt',
                          subsample = 0.8,
                          n_estimators=500,
                          max_depth=3,
                          random_state=42,
                          silent=True,
                          n_jobs=-1,
                          )
clf.fit(X_train, y_train)

ax = lgbm_plt(clf,max_num_features = 20)
fig = ax.figure
fig.set_size_inches(10, 10)

y_pred=clf.predict(X_test)
y_pred_train = clf.predict(X_train)

# print(accuracy_score(y_pred_train, y_train))
# print(accuracy_score(y_pred, y_test))

# cm = confusion_matrix(y_test, y_pred)
# print(cm)

# print(classification_report(y_train, y_pred_train))
# print(classification_report(y_test, y_pred))

print('Light GBM')
print('precision: ' + "{:.3f}".format(precision_score(y_test, y_pred)))
print('recall: ' + "{:.3f}".format(recall_score(y_test, y_pred)))
print('f1-score: ' + "{:.3f}".format(f1_score(y_test, y_pred)))
print('accuracy: ' + "{:.3f}".format(accuracy_score(y_test, y_pred)))

"""XGB"""
xg = XGBClassifier().fit(X_train, y_train)
#sorted_idx = np.argsort(xg.feature_importances_)[::-1]

# feature_names = df.drop('is_prospect',axis=1).columns

# xg.get_booster().feature_names = feature_names

ax = plot_importance(xg,max_num_features = 20)
fig = ax.figure
fig.set_size_inches(10, 10)

y_pred = xg.predict(X_test)
pred_boosting_result = [1 if y_pred[i] >= 0.5 else 0 for i in range(len(y_pred))]

print('xgboost')
print('precision: ' + "{:.3f}".format(precision_score(y_test, pred_boosting_result)))
print('recall: ' + "{:.3f}".format(recall_score(y_test, pred_boosting_result)))
print('f1-score: ' + "{:.3f}".format(f1_score(y_test, pred_boosting_result)))
print('accuracy: ' + "{:.3f}".format(accuracy_score(y_test, pred_boosting_result)))

# cm = confusion_matrix(y_test, pred_boosting_result)
# print(cm)

# Light GBM
# precision: 0.779
# recall: 0.213
# f1-score: 0.335
# accuracy: 0.922

# xgboost
# precision: 0.782
# recall: 0.334
# f1-score: 0.468
# accuracy: 0.930



"""feature engineering"""
df2 = df

df2 = df2.drop('on_boarding_days', axis=1)

# to_drop2 = ['recency','frequency','monetary','label_RFM_segment']
# df2 = df2.drop(to_drop2, axis=1)

# transform_col = [k for k in df.columns if ('txn' in k) or ('spend' in k)]
# for var in transform_col:
#     df2['avg_'+var] = np.where(df2['on_boarding_days']==0,0,df2[var]/df2['on_boarding_days'])
# df2.drop(transform_col, axis=1, inplace=True)

# missing = df2.isnull().sum()/df2.shape[0]
# print(missing[missing > 0])


# fill_col4 = [k for k in df2.columns if ('avg' in k) ]
# df2[fill_col4] = df2[fill_col4].fillna(0)

X = df2.drop('is_prospect', axis=1)
y = df2.is_prospect

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

"""lightGBM"""
clf.fit(X_train, y_train)

ax = lgbm_plt(clf,max_num_features = 20)
fig = ax.figure
fig.set_size_inches(10, 10)

y_pred=clf.predict(X_test)
y_pred_train = clf.predict(X_train)

# print(accuracy_score(y_pred_train, y_train))
# print(accuracy_score(y_pred, y_test))

# cm = confusion_matrix(y_test, y_pred)
# print(cm)

# print(classification_report(y_train, y_pred_train))
# print(classification_report(y_test, y_pred))

print('Light GBM')
print('precision: ' + "{:.3f}".format(precision_score(y_test, y_pred)))
print('recall: ' + "{:.3f}".format(recall_score(y_test, y_pred)))
print('f1-score: ' + "{:.3f}".format(f1_score(y_test, y_pred)))
print('accuracy: ' + "{:.3f}".format(accuracy_score(y_test, y_pred)))

"""XGB"""
xg = XGBClassifier().fit(X_train, y_train)
#sorted_idx = np.argsort(xg.feature_importances_)[::-1]

ax = plot_importance(xg,max_num_features = 20)
fig = ax.figure
fig.set_size_inches(10, 10)

y_pred = xg.predict(X_test)
pred_boosting_result = [1 if y_pred[i] >= 0.5 else 0 for i in range(len(y_pred))]

print('xgboost')
print('precision: ' + "{:.3f}".format(precision_score(y_test, pred_boosting_result)))
print('recall: ' + "{:.3f}".format(recall_score(y_test, pred_boosting_result)))
print('f1-score: ' + "{:.3f}".format(f1_score(y_test, pred_boosting_result)))
print('accuracy: ' + "{:.3f}".format(accuracy_score(y_test, pred_boosting_result)))

# Light GBM
# precision: 0.775
# recall: 0.227
# f1-score: 0.351
# accuracy: 0.922

# xgboost
# precision: 0.776
# recall: 0.332
# f1-score: 0.466
# accuracy: 0.929

"""Oversampling"""
sm = SMOTE(random_state=42)
x_sm, y_sm = sm.fit_resample(X_train, y_train)

clf.fit(x_sm, y_sm)

ax = lgbm_plt(clf,max_num_features = 20)
fig = ax.figure
fig.set_size_inches(10, 10)

y_pred=clf.predict(X_test)
y_pred_train = clf.predict(x_sm)

# print(accuracy_score(y_pred_train, y_train))
# print(accuracy_score(y_pred, y_test))

# cm = confusion_matrix(y_test, y_pred)
# print(cm)

# print(classification_report(y_train, y_pred_train))
# print(classification_report(y_test, y_pred))

print('Light GBM')
print('precision: ' + "{:.3f}".format(precision_score(y_test, y_pred)))
print('recall: ' + "{:.3f}".format(recall_score(y_test, y_pred)))
print('f1-score: ' + "{:.3f}".format(f1_score(y_test, y_pred)))
print('accuracy: ' + "{:.3f}".format(accuracy_score(y_test, y_pred)))

xg = XGBClassifier().fit(x_sm, y_sm)
#sorted_idx = np.argsort(xg.feature_importances_)[::-1]

ax = plot_importance(xg,max_num_features = 20)
fig = ax.figure
fig.set_size_inches(10, 10)

y_pred = xg.predict(X_test)
pred_boosting_result = [1 if y_pred[i] >= 0.5 else 0 for i in range(len(y_pred))]

print('xgboost')
print('precision: ' + "{:.3f}".format(precision_score(y_test, pred_boosting_result)))
print('recall: ' + "{:.3f}".format(recall_score(y_test, pred_boosting_result)))
print('f1-score: ' + "{:.3f}".format(f1_score(y_test, pred_boosting_result)))
print('accuracy: ' + "{:.3f}".format(accuracy_score(y_test, pred_boosting_result)))

# Light GBM
# precision: 0.562
# recall: 0.295
# f1-score: 0.386
# accuracy: 0.913

# xgboost
# precision: 0.742
# recall: 0.331
# f1-score: 0.457
# accuracy: 0.927

"""Undersampling"""
undersm = RandomUnderSampler(sampling_strategy=0.5, random_state=10)
x_us, y_us = undersm.fit_resample(X_train, y_train)

clf.fit(x_us, y_us)

ax = lgbm_plt(clf,max_num_features = 20)
fig = ax.figure
fig.set_size_inches(10, 10)

y_pred=clf.predict(X_test)
y_pred_train = clf.predict(x_us)

print('Light GBM')
print('precision: ' + "{:.3f}".format(precision_score(y_test, y_pred)))
print('recall: ' + "{:.3f}".format(recall_score(y_test, y_pred)))
print('f1-score: ' + "{:.3f}".format(f1_score(y_test, y_pred)))
print('accuracy: ' + "{:.3f}".format(accuracy_score(y_test, y_pred)))

xg = XGBClassifier().fit(x_us, y_us)
#sorted_idx = np.argsort(xg.feature_importances_)[::-1]

ax = plot_importance(xg,max_num_features = 20)
fig = ax.figure
fig.set_size_inches(10, 10)

y_pred = xg.predict(X_test)
pred_boosting_result = [1 if y_pred[i] >= 0.5 else 0 for i in range(len(y_pred))]

print('xgboost')
print('precision: ' + "{:.3f}".format(precision_score(y_test, pred_boosting_result)))
print('recall: ' + "{:.3f}".format(recall_score(y_test, pred_boosting_result)))
print('f1-score: ' + "{:.3f}".format(f1_score(y_test, pred_boosting_result)))
print('accuracy: ' + "{:.3f}".format(accuracy_score(y_test, pred_boosting_result)))

# Light GBM
# precision: 0.381
# recall: 0.674
# f1-score: 0.486
# accuracy: 0.868

# xgboost
# precision: 0.576
# recall: 0.693
# f1-score: 0.629
# accuracy: 0.899

"""get feature importance > 200 & spend_HASE_C"""
# sorted_idx = np.argsort(xg.feature_importances_)[::-1]
# top_features = np.array(x_us.columns)[sorted_idx][0:7]
# top_features = np.insert(top_features,1,'is_prospect')
top_features = ['is_prospect','monetary','spend_WEHK','spend_MNHK','spend_MXHK_nonC','spend_SEHK','spend_IKHK','label_CEM_score_gp','spend_HASE_C']

df3 = df2[top_features]

corr_df = df3.corr().abs()
mask = np.triu(np.ones_like(corr_df, dtype=bool))
tri_df = corr_df.mask(mask)
# remove spend_MNHK

df3 = df3.drop('spend_MNHK',axis=1)

X = df3.drop('is_prospect', axis=1)
y = df3.is_prospect

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

x_us, y_us = undersm.fit_resample(X_train, y_train)

xg = XGBClassifier().fit(x_us, y_us)
#sorted_idx = np.argsort(xg.feature_importances_)[::-1]

ax = plot_importance(xg)
fig = ax.figure
fig.set_size_inches(10, 10)

y_pred = xg.predict(X_test)
pred_boosting_result = [1 if y_pred[i] >= 0.5 else 0 for i in range(len(y_pred))]

print('xgboost')
print('precision: ' + "{:.3f}".format(precision_score(y_test, pred_boosting_result)))
print('recall: ' + "{:.3f}".format(recall_score(y_test, pred_boosting_result)))
print('f1-score: ' + "{:.3f}".format(f1_score(y_test, pred_boosting_result)))
print('accuracy: ' + "{:.3f}".format(accuracy_score(y_test, pred_boosting_result)))

disp = plot_precision_recall_curve(xg, X_test, y_test)
disp.ax_.set_title('Precision-Recall curve:')

cm = confusion_matrix(y_test, pred_boosting_result)
print(cm)

print(classification_report(y_test, pred_boosting_result))

"""final tuning"""
pipeline = Pipeline(steps=[('sampling', undersm), ('class', xg)])
kfold = StratifiedKFold(n_splits=5, random_state=17, shuffle=True)


param_test1 = {
 'xgbclassifier__max_depth':range(3,10,2),
 'xgbclassifier__min_child_weight':range(1,6,2)
}
imba_pipeline = make_pipeline(RandomUnderSampler(sampling_strategy=0.5, random_state=10),
                                  XGBClassifier(
                             learning_rate =0.1,
                             n_estimators=140,
                             max_depth=5,
                             min_child_weight=1,
                             gamma=0,
                             subsample=0.8,
                             colsample_bytree=0.8,
                             objective= 'binary:logistic',
                             nthread=4,
                             scale_pos_weight=1,
                             seed=10, verbose=5))
gsearch1 = GridSearchCV(imba_pipeline, 
 param_grid = param_test1, scoring='recall',n_jobs=4, cv=kfold)
gsearch1.fit(X_train, y_train)
gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_

 # {'xgbclassifier__max_depth': 9, 'xgbclassifier__min_child_weight': 3},
 # 0.604680054204393)
    
param_test2 = {
 'xgbclassifier__max_depth':[8,9,10],
 'xgbclassifier__min_child_weight':[2,2.5,3,3.5,4]
}
imba_pipeline = make_pipeline(RandomUnderSampler(sampling_strategy=0.5, random_state=10),
                                  XGBClassifier(
                             learning_rate =0.1,
                             n_estimators=140,
                             max_depth=5,
                             min_child_weight=1,
                             gamma=0,
                             subsample=0.8,
                             colsample_bytree=0.8,
                             objective= 'binary:logistic',
                             nthread=4,
                             scale_pos_weight=1,
                             seed=10, verbose=5))
gsearch2 = GridSearchCV(imba_pipeline, 
 param_grid = param_test2, scoring='recall',n_jobs=4, cv=kfold)
gsearch2.fit(X_train, y_train)
gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_
 # {'xgbclassifier__max_depth': 10, 'xgbclassifier__min_child_weight': 3},
 # 0.6054219988698746)
 
 
param_test3 = {
 'xgbclassifier__gamma':[i/10.0 for i in range(0,5)]
}
imba_pipeline = make_pipeline(RandomUnderSampler(sampling_strategy=0.5, random_state=10),
                              XGBClassifier(
                             learning_rate =0.1,
                             n_estimators=140,
                             max_depth=10,
                             min_child_weight=3,
                             gamma=0,
                             subsample=0.8,
                             colsample_bytree=0.8,
                             objective= 'binary:logistic',
                             nthread=4,
                             scale_pos_weight=1,
                             seed=10, verbose=5))
gsearch3 = GridSearchCV(imba_pipeline, 
 param_grid = param_test3, scoring='recall',n_jobs=4, cv=kfold)
gsearch3.fit(X_train, y_train)
gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_

param_test4 = {
 'xgbclassifier__subsample':[i/10.0 for i in range(6,10)],
 'xgbclassifier__colsample_bytree':[i/10.0 for i in range(6,10)]
}
imba_pipeline = make_pipeline(RandomUnderSampler(sampling_strategy=0.5, random_state=10),
                              XGBClassifier(
                             learning_rate =0.1,
                             n_estimators=140,
                             max_depth=10,
                             min_child_weight=3,
                             gamma=0,
                             subsample=0.8,
                             colsample_bytree=0.8,
                             objective= 'binary:logistic',
                             nthread=4,
                             scale_pos_weight=1,
                             seed=10, verbose=5))
gsearch4 = GridSearchCV(imba_pipeline, 
 param_grid = param_test4, scoring='recall',n_jobs=4, cv=kfold)
gsearch4.fit(X_train, y_train)
gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_

param_test5 = {
 'xgbclassifier__subsample':[i/100.0 for i in range(75,100,5)],
 'xgbclassifier__colsample_bytree':[i/100.0 for i in range(55,85,5)]
}

imba_pipeline = make_pipeline(RandomUnderSampler(sampling_strategy=0.5, random_state=10),
                              XGBClassifier(
                             learning_rate =0.1,
                             n_estimators=140,
                             max_depth=10,
                             min_child_weight=3,
                             gamma=0,
                             subsample=0.9,
                             colsample_bytree=0.7,
                             objective= 'binary:logistic',
                             nthread=4,
                             scale_pos_weight=1,
                             seed=10, verbose=5))
gsearch5 = GridSearchCV(imba_pipeline, 
 param_grid = param_test5, scoring='recall',n_jobs=4, cv=kfold)
gsearch5.fit(X_train, y_train)
gsearch5.cv_results_, gsearch5.best_params_, gsearch5.best_score_

param_test6 = {
 'xgbclassifier__reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
imba_pipeline = make_pipeline(RandomUnderSampler(sampling_strategy=0.5, random_state=10),
                              XGBClassifier(
 learning_rate =0.1,
 n_estimators=140,
 max_depth=2,
 min_child_weight=0,
 gamma=0,
 subsample=0.45,
 colsample_bytree=0.45,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=10, verbose=5))
gsearch6 = GridSearchCV(imba_pipeline, 
 param_grid = param_test6, scoring='recall',n_jobs=4, cv=kfold)
gsearch6.fit(x_us, y_us)
gsearch6.cv_results_, gsearch6.best_params_, gsearch6.best_score_

param_test7 = {
 'xgbclassifier__reg_alpha':[0.000005, 0.00001, 0.000015, 0.00002]
}
imba_pipeline = make_pipeline(RandomUnderSampler(sampling_strategy=0.5, random_state=10),
                              XGBClassifier(
                             learning_rate =0.1,
                             n_estimators=140,
                             max_depth=2,
                             min_child_weight=0,
                             gamma=0,
                             subsample=0.45,
                             colsample_bytree=0.45,
                             objective= 'binary:logistic',
                             nthread=4,
                             scale_pos_weight=1,
                             seed=10, verbose=5))
gsearch7 = GridSearchCV(imba_pipeline, 
 param_grid = param_test6, scoring='recall',n_jobs=4, cv=kfold)
gsearch7.fit(x_us, y_us)
gsearch7.cv_results_, gsearch7.best_params_, gsearch7.best_score_

kfold = StratifiedKFold(n_splits=5, shuffle=False)
imba_pipeline = make_pipeline(RandomUnderSampler(sampling_strategy=0.5, random_state=10),
                              XGBClassifier(
                             learning_rate =0.01,
                             n_estimators=5000,
                             max_depth=2,
                             min_child_weight=0,
                             gamma=0,
                             subsample=0.45,
                             colsample_bytree=0.45,
                             reg_alpha=1e-5,                               
                             objective= 'binary:logistic',
                             nthread=4,
                             scale_pos_weight=1,
                             seed=10, verbose=5))
cross_val_score(imba_pipeline, X, y, scoring='recall', cv=kfold)