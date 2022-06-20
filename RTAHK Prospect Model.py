# Databricks notebook source
pip install imblearn

# COMMAND ----------

pip install xgboost

# COMMAND ----------

import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve
from imblearn.over_sampling import SMOTE 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBClassifier, plot_importance
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.under_sampling import (RandomUnderSampler, 
                                     ClusterCentroids,
                                     TomekLinks,
                                     NeighbourhoodCleaningRule,
                                     NearMiss)

# COMMAND ----------

df = pd.read_csv('/dbfs/mnt/terence/prospect_model/data_rta.csv')

# COMMAND ----------

df.head()

# COMMAND ----------

df2 = df[(df['segment'] != 'Inactive') & (df['segment'] != 'Lapsed') & (df['segment'] != 'Dormant') & (df['segment'] != 'Occasional') & (df['segment'] != 'Mass')]

# df2 = df[(df['segment'] == 'VIP')]

# COMMAND ----------

df2.shape

# COMMAND ----------

df2['ecom_active'].value_counts()

# COMMAND ----------

df2 = df[['sales_p01', 'sales_p02', 'sales_p03',
            'freq_p01', 'freq_p02', 'freq_p03',
            'cross_banner_p01', 'cross_banner_p02', 'cross_banner_p03',
            'subcategory_p01', 'subcategory_p02', 'subcategory_p03', 
            'lapsed_rfm']]

# COMMAND ----------

x = df2.iloc[:, 3:-1]
y = df2.iloc[:, -1]

# COMMAND ----------

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2,
                                                    random_state=10,
                                                    stratify=y)

# COMMAND ----------

sampler = NearMiss(version=1)
x_rs, y_rs = sampler.fit_resample(x_train, y_train)

# COMMAND ----------

x_train.shape, y_train.shape
x_rs.shape, y_rs.shape

# COMMAND ----------

# fit model no training data
xg = XGBClassifier().fit(x_train, y_train)
# plot feature importance
ax = plot_importance(xg)
fig = ax.figure
fig.set_size_inches(10, 10)

# COMMAND ----------

# y_pred = xg_class.predict(x_test)
# print(y_pred)
y_pred = xg.predict_proba(x_test)

pred_boosting_result = [1 if y_pred[i][1] >= 0.5 else 0 for i in range(len(y_pred))]
print('xgboost')
print('precision: ' + "{:.3f}".format(precision_score(y_test, pred_boosting_result)))
print('recall: ' + "{:.3f}".format(recall_score(y_test, pred_boosting_result)))
print('f1-score: ' + "{:.3f}".format(f1_score(y_test, pred_boosting_result)))
print('accuracy: ' + "{:.3f}".format(accuracy_score(y_test, pred_boosting_result)))

# COMMAND ----------

# fit model no training data
xg_us = XGBClassifier().fit(x_rs, y_rs)
# plot feature importance
ax = plot_importance(xg_us)
fig = ax.figure
fig.set_size_inches(10, 10)

# COMMAND ----------

# y_pred = xg_class.predict(x_test)
# print(y_pred)
y_pred = xg_us.predict_proba(x_test)

pred_boosting_result = [1 if y_pred[i][1] >= 0.5 else 0 for i in range(len(y_pred))]
print('xgboost')
print('precision: ' + "{:.3f}".format(precision_score(y_test, pred_boosting_result)))
print('recall: ' + "{:.3f}".format(recall_score(y_test, pred_boosting_result)))
print('f1-score: ' + "{:.3f}".format(f1_score(y_test, pred_boosting_result)))
print('accuracy: ' + "{:.3f}".format(accuracy_score(y_test, pred_boosting_result)))

# COMMAND ----------

sampler2 = NearMiss(version=2)
x_rs2, y_rs2 = sampler2.fit_resample(x_train, y_train)

# COMMAND ----------

# fit model no training data
xg_us2 = XGBClassifier().fit(x_rs2, y_rs2)
# plot feature importance
ax = plot_importance(xg_us2)
fig = ax.figure
fig.set_size_inches(10, 10)

# COMMAND ----------

y_pred_sm = xg_sm.predict_proba(x_test)
pred_boosting_result_sm = [1 if y_pred_sm[i][1] >= 0.5 else 0 for i in range(len(y_pred_sm))]
print('xgboost_sm')
print('precision: ' + "{:.3f}".format(precision_score(y_test, pred_boosting_result_sm)))
print('recall: ' + "{:.3f}".format(recall_score(y_test, pred_boosting_result_sm)))
print('f1-score: ' + "{:.3f}".format(f1_score(y_test, pred_boosting_result_sm)))
print('accuracy: ' + "{:.3f}".format(accuracy_score(y_test, pred_boosting_result_sm)))

# COMMAND ----------

y_pred_sm = xg_reg_sm.predict(x_test)
pred_boosting_result_sm = [1 if y_pred_sm[i] >= 0.5 else 0 for i in range(len(y_pred_sm))]
print('xgboost_sm')
print('precision: ' + "{:.3f}".format(precision_score(y_test, pred_boosting_result_sm)))
print('recall: ' + "{:.3f}".format(recall_score(y_test, pred_boosting_result_sm)))
print('f1-score: ' + "{:.3f}".format(f1_score(y_test, pred_boosting_result_sm)))
print('accuracy: ' + "{:.3f}".format(accuracy_score(y_test, pred_boosting_result_sm)))

# COMMAND ----------

xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=1)
sm = SMOTE(random_state=1)
pipeline = Pipeline(steps=[('sampling', sm), ('class', xgb1)])
kfold = StratifiedKFold(n_splits=5, random_state=17, shuffle=True)
grid_cv = GridSearchCV(pipeline, scoring='recall', cv=kfold, verbose=5, n_jobs=-1)
# rand_cv = RandomizedSearchCV(pipeline, xgb_grid, scoring='recall', cv=kfold, verbose=5, n_iter=1)
# rand_cv = RandomizedSearchCV(pipeline, xgb_grid, scoring='recall', cv=kfold, verbose=5, n_jobs=-1, n_iter=2)

# COMMAND ----------

rand_cv.fit(x_train, y_train)

# COMMAND ----------

kfold = StratifiedKFold(n_splits=5, shuffle=False)
# imba_pipeline = make_pipeline(SMOTE(random_state=42), 
#                               XGBClassifier(
#  learning_rate =0.1,
#  n_estimators=1000,
#  max_depth=5,
#  min_child_weight=1,
#  gamma=0,
#  subsample=0.8,
#  colsample_bytree=0.8,
#  objective= 'binary:logistic',
#  nthread=4,
#  scale_pos_weight=1,
#  seed=10))
# cross_val_score(imba_pipeline, x_train, y_train, scoring='recall', cv=kfold)

# COMMAND ----------

param_test1 = {
 'xgbclassifier__max_depth':range(3,10,2),
 'xgbclassifier__min_child_weight':range(1,6,2)
}
imba_pipeline = make_pipeline(SMOTE(random_state=42), 
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
gsearch1.fit(x_train, y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

# COMMAND ----------

gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_

# COMMAND ----------

param_test2 = {
 'xgbclassifier__max_depth': [2,3,4],
}
imba_pipeline = make_pipeline(SMOTE(random_state=42), 
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
gsearch2.fit(x_train, y_train)
gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_

# COMMAND ----------

param_test3 = {
 'xgbclassifier__gamma':[i/10.0 for i in range(0,5)]
}
imba_pipeline = make_pipeline(SMOTE(random_state=42), 
                              XGBClassifier(
 learning_rate =0.1,
 n_estimators=140,
 max_depth=2,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=10, verbose=5))
gsearch3 = GridSearchCV(imba_pipeline, 
 param_grid = param_test3, scoring='recall',n_jobs=4, cv=kfold)
gsearch3.fit(x_train, y_train)
gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_

# COMMAND ----------

param_test4 = {
 'xgbclassifier__subsample':[i/10.0 for i in range(6,10)],
 'xgbclassifier__colsample_bytree':[i/10.0 for i in range(6,10)]
}
imba_pipeline = make_pipeline(SMOTE(random_state=42), 
                              XGBClassifier(
 learning_rate =0.1,
 n_estimators=140,
 max_depth=2,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=10, verbose=5))
gsearch4 = GridSearchCV(imba_pipeline, 
 param_grid = param_test4, scoring='recall',n_jobs=4, cv=kfold)
gsearch4.fit(x_train, y_train)
gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_

# COMMAND ----------

param_test5 = {
 'xgbclassifier__subsample':[i/100.0 for i in range(45,75,5)],
 'xgbclassifier__colsample_bytree':[i/100.0 for i in range(45,75,5)]
}

imba_pipeline = make_pipeline(SMOTE(random_state=42), 
                              XGBClassifier(
 learning_rate =0.1,
 n_estimators=140,
 max_depth=2,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=10, verbose=5))
gsearch5 = GridSearchCV(imba_pipeline, 
 param_grid = param_test5, scoring='recall',n_jobs=4, cv=kfold)
gsearch5.fit(x_train, y_train)
gsearch5.cv_results_, gsearch5.best_params_, gsearch5.best_score_

# COMMAND ----------

param_test6 = {
 'xgbclassifier__reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
imba_pipeline = make_pipeline(SMOTE(random_state=42), 
                              XGBClassifier(
 learning_rate =0.1,
 n_estimators=140,
 max_depth=2,
 min_child_weight=1,
 gamma=0,
 subsample=0.45,
 colsample_bytree=0.45,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=10, verbose=5))
gsearch6 = GridSearchCV(imba_pipeline, 
 param_grid = param_test6, scoring='recall',n_jobs=4, cv=kfold)
gsearch6.fit(x_train, y_train)
gsearch6.cv_results_, gsearch6.best_params_, gsearch6.best_score_

# COMMAND ----------

param_test7 = {
 'xgbclassifier__reg_alpha':[0.05, 0.1, 0.15, 0.2]
}
imba_pipeline = make_pipeline(SMOTE(random_state=42), 
                              XGBClassifier(
 learning_rate =0.1,
 n_estimators=140,
 max_depth=2,
 min_child_weight=1,
 gamma=0,
 subsample=0.45,
 colsample_bytree=0.45,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=10, verbose=5))
gsearch7 = GridSearchCV(imba_pipeline, 
 param_grid = param_test6, scoring='recall',n_jobs=4, cv=kfold)
gsearch7.fit(x_train, y_train)
gsearch7.cv_results_, gsearch7.best_params_, gsearch7.best_score_

# COMMAND ----------

kfold = StratifiedKFold(n_splits=5, shuffle=False)
imba_pipeline = make_pipeline(SMOTE(random_state=42), 
                              XGBClassifier(
 learning_rate =0.01,
 n_estimators=5000,
 max_depth=2,
 min_child_weight=1,
 gamma=0,
 subsample=0.45,
 colsample_bytree=0.45,
 reg_alpha=0.1,                               
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=10, verbose=5))
cross_val_score(imba_pipeline, x, y, scoring='recall', cv=kfold)

# COMMAND ----------


