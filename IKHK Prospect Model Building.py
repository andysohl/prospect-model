# Databricks notebook source
# MAGIC %md
# MAGIC # IKEA Prospect Model
# MAGIC Building a model to predict the probability that a prospect member will make a purchase at IKEA
# MAGIC 
# MAGIC (20211220 UPDATE) Training the model with rolling historical data, instead of 6 month total
# MAGIC 
# MAGIC (20211222 UPDATE) Training the model with rolling weekly data, instead of monthly data
# MAGIC 
# MAGIC (20211223 UPDATE) Training the model with 3 month worth rolling weekly data
# MAGIC 
# MAGIC (20220106 UPDATE) Updated rules and data for LWCF

# COMMAND ----------

# MAGIC %md
# MAGIC #### Import libraries

# COMMAND ----------

# Import MLFlow and enable autologging for xgboost
import mlflow.xgboost
mlflow.xgboost.autolog()

# COMMAND ----------

# Import MLFlow and enable autologging for scikit-learn
import mlflow.sklearn
mlflow.sklearn.autolog()

# COMMAND ----------

import mlflow
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from imblearn.over_sampling import SMOTE 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier, plot_importance
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.under_sampling import (RandomUnderSampler, 
                                     CondensedNearestNeighbour,
                                     ClusterCentroids,
                                     TomekLinks,
                                     NeighbourhoodCleaningRule,
                                     NearMiss)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Import and EDA

# COMMAND ----------

df = pd.read_csv('/dbfs/mnt/04publish_test/rex/ikea_data_20220104.csv')

# COMMAND ----------

df.head(10)

# COMMAND ----------

df.tail(10)

# COMMAND ----------

df.shape

# COMMAND ----------

df['overall_segment'].value_counts()

# COMMAND ----------

df['ikea_segment'].value_counts()

# COMMAND ----------

# Only consider members that are in the IKEA RFM segments "Prospect"
df2 = df[(df['ikea_segment'] == 'Prospect')]
# & (df['segment'] != 'Occasional')  & (df['segment'] != 'Bulk buyers')
# df2 = df[(df['segment'] == 'VIP')]

# COMMAND ----------

df2.shape

# COMMAND ----------

df2['ikea_segment'].value_counts()

# COMMAND ----------

df2['ikea_next_segment'].value_counts()

# COMMAND ----------

df2['overall_segment'].value_counts()

# COMMAND ----------

# Remove members classified as inactive / lapsed / dormant in the overall RFM model
df3 = df2[(df2['overall_segment'] != 'Inactive') & (df2['overall_segment'] != 'Lapsed') & (df2['overall_segment'] != 'Dormant') & (df2['overall_segment'] != 'New') & (df['overall_segment'] != 'Bulk buyers')]
# & (df['segment'] != 'Occasional')  & (df['segment'] != 'Bulk buyers')
# df2 = df[(df['segment'] == 'VIP')]

# COMMAND ----------

df3.shape

# COMMAND ----------

df3['overall_segment'].value_counts()

# COMMAND ----------

df3['ikea_segment'].value_counts()

# COMMAND ----------

df3['ikea_next_segment'].value_counts(normalize=True)

# COMMAND ----------

# Create a new column as target, indicating a member has changed its RFM status from "Prospect" to "New" 
df3['ground_truth'] = np.where(df3.ikea_next_segment == "New", 1, 0)

# COMMAND ----------

df3.head()

# COMMAND ----------

df3[df3['ground_truth'] == 1]

# COMMAND ----------

df3.dtypes

# COMMAND ----------

a= list(df3.columns)
print(a)

# COMMAND ----------

df3['overall_segment'] = df3['overall_segment'].astype("category")
df3['ikea_segment'] = df3['ikea_segment'].astype("category")
df3['ikea_next_segment'] = df3['ikea_next_segment'].astype("category")
base_var_name_list = ['freq_mnhk_t', 'freq_sehk_t', 'freq_wehk_t', 'sales_wehk_t', 'sales_sehk_t', 'sales_mnhk_t']
for i in range(13):
  for base in base_var_name_list:
    var_name = base + str(i+1)
    df3[var_name] = df3[var_name].astype("int64")
df3['LWCF'] = df3['LWCF'].astype("int64")
df3['babynkidpdt'] = df3['babynkidpdt'].astype("int64")
df3['HASE_Family'] = df3['HASE_Family'].astype("int64")
df3['Chi_Restaurant'] = df3['Chi_Restaurant'].astype("int64")
df3['WE_Keans'] = df3['WE_Keans'].astype("int64")
df3.dtypes

# COMMAND ----------

df4 = df3.drop(['ikea_segment', 'ikea_next_segment'], axis=1)
df4.dtypes

# COMMAND ----------

def label_LWCF (row):
    if (row['WE_Keans'] == 1) or (row['babynkidpdt'] == 1) or (row['HASE_Family'] == 1) or (row['Chi_Restaurant'] == 1):
        return 1
    else:
        return 0

# COMMAND ----------

df4.apply(lambda row: label_LWCF(row), axis=1)

# COMMAND ----------

df4['LWCF_2'] = df4.apply(lambda row: label_LWCF(row), axis=1)
df4['LWCF_2'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Split data into training and testing set 

# COMMAND ----------

# x_all = df4[['sales_wehk_t1', 'sales_mnhk_t1', 'sales_wehk_t2', 'sales_mnhk_t2', 'sales_wehk_t3', 'sales_mnhk_t3', 'sales_wehk_t4', 'sales_mnhk_t4', 'sales_wehk_t5', 'sales_mnhk_t5', 'sales_wehk_t6', 'sales_mnhk_t6', 'catchment', 'comm_open_rate', 'living_with_child', 'view_per_day']]
# x_all = df4[['beverage_sales1', 'toilet_paper_sales1', 'icecream_sales1', 'frozen_food_sales1', 'soft_drink_sales1', 'snack_sales1', 'baby_product_sales1', 'household_sales1', 'beverage_sales2', 'toilet_paper_sales2', 'icecream_sales2', 'frozen_food_sales2', 'soft_drink_sales2', 'snack_sales2', 'baby_product_sales2', 'household_sales2', 'beverage_sales3', 'toilet_paper_sales3', 'icecream_sales3', 'frozen_food_sales3', 'soft_drink_sales3', 'snack_sales3', 'baby_product_sales3', 'household_sales3', 'beverage_sales4', 'toilet_paper_sales4', 'icecream_sales4', 'frozen_food_sales4', 'soft_drink_sales4', 'snack_sales4', 'baby_product_sales4', 'household_sales4', 'beverage_sales5', 'toilet_paper_sales5', 'icecream_sales5', 'frozen_food_sales5', 'soft_drink_sales5', 'snack_sales5', 'baby_product_sales5', 'household_sales5', 'beverage_sales6', 'toilet_paper_sales6', 'icecream_sales6', 'frozen_food_sales6', 'soft_drink_sales6', 'snack_sales6', 'baby_product_sales6', 'household_sales6', 'beverage_sales7', 'toilet_paper_sales7', 'icecream_sales7', 'frozen_food_sales7', 'soft_drink_sales7', 'snack_sales7', 'baby_product_sales7', 'household_sales7', 'beverage_sales8', 'toilet_paper_sales8', 'icecream_sales8', 'frozen_food_sales8', 'soft_drink_sales8', 'snack_sales8', 'baby_product_sales8', 'household_sales8', 'beverage_sales9', 'toilet_paper_sales9', 'icecream_sales9', 'frozen_food_sales9', 'soft_drink_sales9', 'snack_sales9', 'baby_product_sales9', 'household_sales9', 'beverage_sales10', 'toilet_paper_sales10', 'icecream_sales10', 'frozen_food_sales10', 'soft_drink_sales10', 'snack_sales10', 'baby_product_sales10', 'household_sales10', 'beverage_sales11', 'toilet_paper_sales11', 'icecream_sales11', 'frozen_food_sales11', 'soft_drink_sales11', 'snack_sales11', 'baby_product_sales11', 'household_sales11', 'beverage_sales12', 'toilet_paper_sales12', 'icecream_sales12', 'frozen_food_sales12', 'soft_drink_sales12', 'snack_sales12', 'baby_product_sales12', 'household_sales12', 'beverage_sales13', 'toilet_paper_sales13', 'icecream_sales13', 'frozen_food_sales13', 'soft_drink_sales13', 'snack_sales13', 'baby_product_sales13', 'household_sales13', 'comm_open_rate', 'view_per_day']]
# , 'LWCF'
x_all = df4[['view_per_day', 'comm_open_rate', 'LWCF_2']]
y_all = df4['ground_truth']
#  'imf_diaper_sales', 'beverage_sales', 'toilet_paper_sales', 'icecream_sales', 'frozen_food_sales', 'soft_drink_sales', 'snack_sales', 'baby_product_sales', 'household_sales',

# COMMAND ----------

print(df4['WE_Keans'].value_counts())
print(df4['babynkidpdt'].value_counts())
print(df4['HASE_Family'].value_counts())
print(df4['Chi_Restaurant'].value_counts())
print(df4['LWCF_2'].value_counts())

# COMMAND ----------

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all,
                                                    test_size=0.2,
                                                    random_state=10,
                                                    stratify=y_all)

# COMMAND ----------

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# COMMAND ----------

x_train.dtypes

# COMMAND ----------

# fit model no training data
xg = XGBClassifier(objective='binary:logistic', use_label_encoder=False, n_estimators=100).fit(x_train._get_numeric_data(), np.ravel(y_train))
# plot feature importance
ax = plot_importance(xg)
fig = ax.figure
fig.set_size_inches(10, 10)

# COMMAND ----------

xg.feature_importances_

# COMMAND ----------

y_pred = xg.predict_proba(x_test._get_numeric_data())
pred_boosting_result = [1 if y_pred[i][1] >= 0.5 else 0 for i in range(len(y_pred))]
# y_pred = xg.predict(x_test._get_numeric_data())
# pred_boosting_result = y_pred
print('xgboost')
print('precision: ' + "{:.3f}".format(precision_score(y_test._get_numeric_data(), pred_boosting_result)))
print('recall: ' + "{:.3f}".format(recall_score(y_test._get_numeric_data(), pred_boosting_result)))
print('f1-score: ' + "{:.3f}".format(f1_score(y_test._get_numeric_data(), pred_boosting_result)))
print('accuracy: ' + "{:.3f}".format(accuracy_score(y_test._get_numeric_data(), pred_boosting_result)))
print('confusion matrix:')
print(confusion_matrix(y_test._get_numeric_data(), pred_boosting_result))
print('ROC AUC score: ' + "{:.3f}".format(roc_auc_score(y_test._get_numeric_data(), pred_boosting_result)))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Oversampling

# COMMAND ----------

sm = SMOTE(random_state=123, sampling_strategy=0.1)
x_sm, y_sm = sm.fit_resample(x_train._get_numeric_data(), np.ravel(y_train))
# fit model no training data
xg_sm = XGBClassifier(objective='binary:logistic', use_label_encoder=False, n_estimators=100).fit(x_sm, y_sm)
# plot feature importance
ax = plot_importance(xg_sm)
fig = ax.figure
fig.set_size_inches(10, 10)

# COMMAND ----------

y_pred = xg_sm.predict_proba(x_test._get_numeric_data())
pred_boosting_result = [1 if y_pred[i][1] >= 0.5 else 0 for i in range(len(y_pred))]
# y_pred = xg.predict(x_test._get_numeric_data())
# pred_boosting_result = y_pred
print('xgboost')
print('precision: ' + "{:.3f}".format(precision_score(y_test._get_numeric_data(), pred_boosting_result)))
print('recall: ' + "{:.3f}".format(recall_score(y_test._get_numeric_data(), pred_boosting_result)))
print('f1-score: ' + "{:.3f}".format(f1_score(y_test._get_numeric_data(), pred_boosting_result)))
print('accuracy: ' + "{:.3f}".format(accuracy_score(y_test._get_numeric_data(), pred_boosting_result)))
print('confusion matrix:')
print(confusion_matrix(y_test._get_numeric_data(), pred_boosting_result))
print('ROC AUC score: ' + "{:.3f}".format(roc_auc_score(y_test._get_numeric_data(), pred_boosting_result)))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Undersampling

# COMMAND ----------

# undersample = RandomUnderSampler(sampling_strategy='majority', random_state=123)
undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=10)
x_us, y_us = undersample.fit_resample(x_train._get_numeric_data(), np.ravel(y_train))
# fit model no training data
xg_us = XGBClassifier(objective='binary:logistic', use_label_encoder=False, n_estimators=100, max_depth=10, learning_rate=0.1, colsample_bytree=1, subsample=1).fit(x_us, y_us)
# plot feature importance
ax = plot_importance(xg_us)
fig = ax.figure
fig.set_size_inches(10, 10)

# COMMAND ----------

y_pred = xg_us.predict_proba(x_test._get_numeric_data())
pred_boosting_result = [1 if y_pred[i][1] >= 0.5 else 0 for i in range(len(y_pred))]
# y_pred = xg.predict(x_test._get_numeric_data())
# pred_boosting_result = y_pred
print('xgboost')
print('precision: ' + "{:.3f}".format(precision_score(y_test._get_numeric_data(), pred_boosting_result)))
print('recall: ' + "{:.3f}".format(recall_score(y_test._get_numeric_data(), pred_boosting_result)))
print('f1-score: ' + "{:.3f}".format(f1_score(y_test._get_numeric_data(), pred_boosting_result)))
print('accuracy: ' + "{:.3f}".format(accuracy_score(y_test._get_numeric_data(), pred_boosting_result)))

# COMMAND ----------

confusion_matrix(y_test._get_numeric_data(), pred_boosting_result)

# COMMAND ----------

roc_auc_score(y_test._get_numeric_data(), pred_boosting_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Oversample then undersample

# COMMAND ----------

sm = SMOTE(sampling_strategy=0.1,random_state=123)
x_sm, y_sm = sm.fit_resample(x_train._get_numeric_data(), np.ravel(y_train))
undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=123)
x_us, y_us = undersample.fit_resample(x_sm._get_numeric_data(), np.ravel(y_sm))
# fit model no training data
xg_us = XGBClassifier(objective='binary:logistic', use_label_encoder=False, n_estimators=100, max_depth=3).fit(x_us, y_us)
# , max_depth=10, learning_rate=0.01, colsample_bytree=1, subsample=1
# plot feature importance
ax = plot_importance(xg_us)
fig = ax.figure
fig.set_size_inches(10, 10)

# COMMAND ----------

y_pred = xg_us.predict_proba(x_test._get_numeric_data())
pred_boosting_result = [1 if y_pred[i][1] >= 0.5 else 0 for i in range(len(y_pred))]
# y_pred = xg.predict(x_test._get_numeric_data())
# pred_boosting_result = y_pred
print('xgboost')
print('precision: ' + "{:.3f}".format(precision_score(y_test._get_numeric_data(), pred_boosting_result)))
print('recall: ' + "{:.3f}".format(recall_score(y_test._get_numeric_data(), pred_boosting_result)))
print('f1-score: ' + "{:.3f}".format(f1_score(y_test._get_numeric_data(), pred_boosting_result)))
print('accuracy: ' + "{:.3f}".format(accuracy_score(y_test._get_numeric_data(), pred_boosting_result)))
print('confusion matrix:')
print(confusion_matrix(y_test._get_numeric_data(), pred_boosting_result))
print('ROC AUC score: ' + "{:.3f}".format(roc_auc_score(y_test._get_numeric_data(), pred_boosting_result)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hyperparameter Tuning

# COMMAND ----------

kfold = StratifiedKFold(n_splits=5, shuffle=False)

# COMMAND ----------

param_test1 = {
#  'xgbclassifier__n_estimators': [100,500,1000],
#  'xgbclassifier__max_depth':range(3,10,2),
#  'xgbclassifier__min_child_weight':range(1,6,2),
 'xgbclassifier__gamma':[i/10.0 for i in range(0,5)],
#  'xgbclassifier__subsample':[i/10.0 for i in range(6,10)]
 'xgbclassifier__colsample_bytree':[i/10.0 for i in range(6,10)],
 'xgbclassifier__reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
imba_pipeline = make_pipeline(RandomUnderSampler(sampling_strategy='majority', random_state=123), 
                              XGBClassifier(
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=123, verbose=5,
 max_depth=3, n_estimators=100, subsample=0.6))
#                                 colsample_bytree=0.6, reg_alpha=0.1))
gsearch1 = GridSearchCV(imba_pipeline, 
 param_grid = param_test1, scoring='f1',n_jobs=4, cv=kfold)
gsearch1.fit(x_all, y_all)
gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_

# COMMAND ----------

y_pred = gsearch1.predict_proba(x_test._get_numeric_data())
pred_boosting_result = [1 if y_pred[i][1] >= 0.5 else 0 for i in range(len(y_pred))]
# y_pred = xg.predict(x_test._get_numeric_data())
# pred_boosting_result = y_pred
print('xgboost')
print('precision: ' + "{:.3f}".format(precision_score(y_test._get_numeric_data(), pred_boosting_result)))
print('recall: ' + "{:.3f}".format(recall_score(y_test._get_numeric_data(), pred_boosting_result)))
print('f1-score: ' + "{:.3f}".format(f1_score(y_test._get_numeric_data(), pred_boosting_result)))
print('accuracy: ' + "{:.3f}".format(accuracy_score(y_test._get_numeric_data(), pred_boosting_result)))
print('confusion matrix:')
print(confusion_matrix(y_test._get_numeric_data(), pred_boosting_result))
print('ROC AUC score: ' + "{:.3f}".format(roc_auc_score(y_test._get_numeric_data(), pred_boosting_result)))

# COMMAND ----------

undersample = RandomUnderSampler(sampling_strategy='majority', random_state=123)
# undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=10)
x_us, y_us = undersample.fit_resample(x_train._get_numeric_data(), np.ravel(y_train))
# fit model no training data
xg_us = XGBClassifier(objective='binary:logistic', use_label_encoder=False, n_estimators=100, max_depth=3, colsample_bytree=0.6, reg_alpha=1, subsample=0.6, gamma=0.4).fit(x_us, y_us)
# plot feature importance
# ax = plot_importance(xg_us)
# fig = ax.figure
# fig.set_size_inches(10, 10)

# COMMAND ----------

y_pred = xg_us.predict_proba(x_test._get_numeric_data())
pred_boosting_result = [1 if y_pred[i][1] >= 0.5 else 0 for i in range(len(y_pred))]
# y_pred = xg.predict(x_test._get_numeric_data())
# pred_boosting_result = y_pred
print('xgboost')
print('precision: ' + "{:.3f}".format(precision_score(y_test._get_numeric_data(), pred_boosting_result)))
print('recall: ' + "{:.3f}".format(recall_score(y_test._get_numeric_data(), pred_boosting_result)))
print('f1-score: ' + "{:.3f}".format(f1_score(y_test._get_numeric_data(), pred_boosting_result)))
print('accuracy: ' + "{:.3f}".format(accuracy_score(y_test._get_numeric_data(), pred_boosting_result)))
print('confusion matrix:')
print(confusion_matrix(y_test._get_numeric_data(), pred_boosting_result))
print('ROC AUC score: ' + "{:.3f}".format(roc_auc_score(y_test._get_numeric_data(), pred_boosting_result)))

# COMMAND ----------

import shap

# COMMAND ----------

explainer = shap.TreeExplainer(xg_us)
shap_values = explainer.shap_values(x_all)

# COMMAND ----------

shap.summary_plot(shap_values, x_all, plot_type="bar")

# COMMAND ----------

shap.summary_plot(shap_values, x_all)

# COMMAND ----------

shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[2951313])

# COMMAND ----------

y_pred = xg_us.predict_proba(x_test._get_numeric_data())
y_pred[888]

# COMMAND ----------

x_all.shape

# COMMAND ----------


