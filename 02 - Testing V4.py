# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Testing V3
# MAGIC Get testing data (member who made their purchase in a certain time period) and use our model to see how many of them it can capture.
# MAGIC 
# MAGIC Find out how many member made their first purchase at IKEA in March 2022, then put their txn history (profile) into the model and compute a similarity distance. Get the most similar N members according to the model and see how many are the same with the actual member set.
# MAGIC 
# MAGIC ### V3 Changes
# MAGIC Try to do table operations on databricks (spark sql) instead of on synapse

# COMMAND ----------

import time
time.sleep(10000)

# COMMAND ----------

# MAGIC %run /Users/trncetou@dairy-farm.com.hk/helper_cls

# COMMAND ----------

# MAGIC %run /Users/trncetou@dairy-farm.com.hk/helper_p_cls

# COMMAND ----------

helper1 = Helper_p()  # For Reading live data from Production 
helper2 = Helper()  # For Writng data into different environment 

# COMMAND ----------

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from pyspark.sql.functions import *
from datetime import datetime, timedelta, date
import pytz
import pyspark.sql.functions as F
from sklearn.neighbors import NearestNeighbors
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import pickle
import os

# COMMAND ----------

# Connect with the database
password=dbutils.secrets.get(scope = "DATALAKE-PD", key = "synapse-pd01-dfsg-datafactory-pwd")
url = "jdbc:sqlserver://sql-gpdl01-pd-sea-df-pd-sql01.database.windows.net:1433;database=sqldwh-gpdl01-pd-sea-dl01-sqldwh01;user=dfsg_datafactory@sql-gpdl01-pd-sea-df-pd-sql01;password="+password+";encrypt=true;trustServerCertificate=true;"

# COMMAND ----------

folder_path = "/dbfs/mnt/rex/ikea_prospect_v2/"

# COMMAND ----------

column_list = pd.read_csv("/dbfs/mnt/rex/ikea_prospect_v2/final_df_column_list_0620.csv", index_col=None)
column_list.iloc[0,0] = 'MEMBER_ID'
column_list

# COMMAND ----------

df_final_empty = pd.DataFrame(columns=column_list['0'].tolist())
df_final_empty

# COMMAND ----------

# Set up important dates for setting time period
# first purchase between 
first_purchase_start_date = '2022-03-01'
first_purchase_end_date = '2022-03-31'
first_purchase_start_date = datetime.strptime(first_purchase_start_date, "%Y-%m-%d")
first_purchase_end_date = datetime.strptime(first_purchase_end_date, "%Y-%m-%d")
txn_hist_start_date = (first_purchase_start_date - relativedelta(months=1)).strftime("%Y-%m-%d")
txn_hist_end_date = (first_purchase_end_date - relativedelta(months=1)).strftime("%Y-%m-%d")
first_purchase_start_date = first_purchase_start_date.strftime("%Y-%m-%d")
first_purchase_end_date = first_purchase_end_date.strftime("%Y-%m-%d")

print(first_purchase_start_date)
print(first_purchase_end_date)
print(txn_hist_start_date)
print(txn_hist_end_date)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sample members for testing (500K, test 100K members everytime)

# COMMAND ----------

df_member_sample = pd.read_csv(folder_path + "testing_member_list_part4.csv", index_col=0)
df_member_sample

# COMMAND ----------

df_final_empty['MEMBER_ID'] = df_member_sample['MEMBER_ID']
df_final_empty

# COMMAND ----------

# MAGIC %md
# MAGIC ### WEHK testing data

# COMMAND ----------

# WEHK
test_wehk_sql = """
select  a.MEMBER_ID
      , c.department_no
      , c.department_name
      , sum(b.final_unit_price) as wehk_sales
      , count(distinct a.TRANSACTION_EXTERNAL_REFERENCE) as wehk_freq
      
FROM [PUBLISH].[ANFIELD_LOY_SALES_METRICS] a

inner join [ANALYSE].[ANFIELD_TX_TRANSACTION_PRODUCT_INFO] b 
on a.TRANSACTION_EXTERNAL_REFERENCE = b.APP_TRANSACTION_ID

inner join [ANALYSE].[HKWE_MD_ITEM] c 
on b.product_item_external_reference = concat('WEHK-', c.item_no)

WHERE a.SPONSOR_BUSINESS_UNIT = 'WEHK'
and a.TRANSACTION_RETAIL_VALUE > 0 
and b.FINAL_UNIT_PRICE > 0
and a.[TIME_PERIOD] BETWEEN '{0}' AND '{1}' -- get txn data between the time period
GROUP BY a.member_id, c.department_no, c.department_name
"""

# COMMAND ----------

# Send the query to data lake, retrieve the result, and save it as dataframe, then CSV file
sdf_wehk = (spark.read
          .format("jdbc")
          .option("url", url)
          .option("query", test_wehk_sql.format(txn_hist_start_date, txn_hist_end_date))
          .load()
            )
display(sdf_wehk)
# df = df_spark.toPandas()
# if not os.path.exists(folder_path + "ikea_first_purchase_limited_member_list_0609.csv"):
#     filename = folder_path + "ikea_first_purchase_limited_member_list_0609.csv"
#     df.to_csv(filename, index=False)
# df

# COMMAND ----------

df_wehk = sdf_wehk.toPandas()
df_wehk['MEMBER_ID'] = pd.to_numeric(df_wehk['MEMBER_ID'])
df_wehk_join = df_wehk.merge(df_member_sample, on="MEMBER_ID", how="inner")
df_wehk_join

# COMMAND ----------

df_wehk_pivot = df_wehk_join.pivot(index='MEMBER_ID', columns='department_name', values=['wehk_freq', 'wehk_sales']).reset_index()
df_wehk_pivot.columns = [f'{i}_{j}' if j != '' else f'{i}' for i, j in df_wehk_pivot.columns]
df_wehk_pivot

# COMMAND ----------

# MAGIC %md
# MAGIC ### SEHK

# COMMAND ----------

# SEHK
test_sehk_sql = """
select  a.MEMBER_ID
      , c.category_name
      , sum(b.final_unit_price) as sehk_sales
      , count(distinct a.TRANSACTION_EXTERNAL_REFERENCE) as sehk_freq
      
FROM [PUBLISH].[ANFIELD_LOY_SALES_METRICS] a

inner join [ANALYSE].[ANFIELD_TX_TRANSACTION_PRODUCT_INFO] b 
on a.TRANSACTION_EXTERNAL_REFERENCE = b.APP_TRANSACTION_ID

inner join [ANALYSE].[HKSE_MD_ITEM] c 
on substring(b.PRODUCT_ITEM_EXTERNAL_REFERENCE,6,9) = c.product_id

WHERE a.SPONSOR_BUSINESS_UNIT = 'SEHK'
and a.TRANSACTION_RETAIL_VALUE > 0 
and b.FINAL_UNIT_PRICE > 0
and a.[TIME_PERIOD] BETWEEN '{0}' AND '{1}'
GROUP BY a.member_id, c.category_name
"""

# COMMAND ----------

# Send the query to data lake, retrieve the result, and save it as dataframe, then CSV file
sdf_sehk = (spark.read
          .format("jdbc")
          .option("url", url)
          .option("query", test_sehk_sql.format(txn_hist_start_date, txn_hist_end_date))
          .load()
            )
display(sdf_sehk)

# COMMAND ----------

df_sehk = sdf_sehk.toPandas()
df_sehk['MEMBER_ID'] = pd.to_numeric(df_sehk['MEMBER_ID'])
df_sehk_join = df_sehk.merge(df_member_sample, on="MEMBER_ID", how="inner")
df_sehk_join

# COMMAND ----------

df_sehk_pivot = df_sehk_join.pivot(index='MEMBER_ID', columns='category_name', values=['sehk_freq', 'sehk_sales']).reset_index()
df_sehk_pivot.columns = [f'{i}_{j}' if j != '' else f'{i}' for i, j in df_sehk_pivot.columns]
df_sehk_pivot

# COMMAND ----------

# MAGIC %md
# MAGIC ### MNHK

# COMMAND ----------

# MNHK
test_mnhk_sql = """
select  a.MEMBER_ID
      , c.bunit_name
      , sum(b.final_unit_price) as mnhk_sales
      , count(distinct a.TRANSACTION_EXTERNAL_REFERENCE) as mnhk_freq
      
FROM [PUBLISH].[ANFIELD_LOY_SALES_METRICS] a

inner join [ANALYSE].[ANFIELD_TX_TRANSACTION_PRODUCT_INFO] b 
on a.TRANSACTION_EXTERNAL_REFERENCE = b.APP_TRANSACTION_ID

inner join [ANALYSE].[HKMN_MD_ITEM] c 
on substring(b.PRODUCT_ITEM_EXTERNAL_REFERENCE,6,6) = c.product_id

WHERE a.SPONSOR_BUSINESS_UNIT = 'MNHK'
and a.TRANSACTION_RETAIL_VALUE > 0 
and b.FINAL_UNIT_PRICE > 0
and a.[TIME_PERIOD] BETWEEN '{0}' AND '{1}'
GROUP BY a.member_id, c.bunit_name
"""

# COMMAND ----------

# Send the query to data lake, retrieve the result, and save it as dataframe, then CSV file
sdf_mnhk = (spark.read
          .format("jdbc")
          .option("url", url)
          .option("query", test_mnhk_sql.format(txn_hist_start_date, txn_hist_end_date))
          .load()
            )
display(sdf_mnhk)

# COMMAND ----------

df_mnhk = sdf_mnhk.toPandas()
df_mnhk['MEMBER_ID'] = pd.to_numeric(df_mnhk['MEMBER_ID'])
df_mnhk_join = df_mnhk.merge(df_member_sample, on="MEMBER_ID", how="inner")
df_mnhk_join

# COMMAND ----------

df_mnhk_pivot = df_mnhk_join.pivot(index='MEMBER_ID', columns='bunit_name', values=['mnhk_freq', 'mnhk_sales']).reset_index()
df_mnhk_pivot.columns = [f'{i}_{j}' if j != '' else f'{i}' for i, j in df_mnhk_pivot.columns]
df_mnhk_pivot

# COMMAND ----------

df_final = df_wehk_pivot.merge(df_sehk_pivot, on='MEMBER_ID', how='outer')
df_final = df_final.merge(df_mnhk_pivot, on='MEMBER_ID', how='outer')
df_final = df_final.merge(df_final_empty, on='MEMBER_ID', how='outer', suffixes=('', '_y'))
df_final = df_final.drop(df_final.filter(regex='_y$').columns, axis=1)
df_final = df_final.reindex(columns=column_list['0'].tolist())
df_final

# COMMAND ----------

df_final.iloc[:,1:].isnull().all(axis=1).value_counts()

# COMMAND ----------

df_final.isnull().all().value_counts()

# COMMAND ----------

df_final = df_final.set_index("MEMBER_ID")
df_final = df_final.dropna(axis=0, how='all')
df_final = df_final.reset_index()
df_final

# COMMAND ----------

df_final.iloc[:,1:].isnull().all(axis=1).value_counts()

# COMMAND ----------

df_final.isnull().all().value_counts()

# COMMAND ----------

# test_final_df = test_final_df.dropna(axis=1, how='all')
df_final = df_final.fillna(0)
df_final

# COMMAND ----------

list(set(column_list['0'].tolist()) ^ set(df_final.columns))

# COMMAND ----------

df_final = df_final.drop(columns=list(set(column_list['0'].tolist()) ^ set(df_final.columns)))
df_final

# COMMAND ----------

df_final.to_csv(folder_path + "testing_data_final_0620_part5.csv")

# COMMAND ----------

df_final = pd.read_csv(folder_path + "testing_data_final_0620_part5.csv", index_col=0)
df_final

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculate distances

# COMMAND ----------

# Load the model
knn = pickle.load(open('/dbfs/mnt/rex/ikea_prospect_v2/knn_model_retail_0620.pkl', 'rb'))

# COMMAND ----------

test_array = df_final.iloc[:,1:].to_numpy()
test_array

# COMMAND ----------

dist, ind = knn.kneighbors(test_array, 10, return_distance=True)

# COMMAND ----------

dist.shape

# COMMAND ----------

ind.shape

# COMMAND ----------

mean_dist_list = np.mean(dist, axis=1)
mean_dist_list.shape

# COMMAND ----------

dist_df = pd.DataFrame(mean_dist_list)
dist_df

# COMMAND ----------

dist_df.describe()

# COMMAND ----------

import matplotlib.pyplot as plt

# fig, ax = plt.subplots(figsize = (10,10))
dist_df.plot(kind = "hist", density = True, bins=100)
plt.show()

# COMMAND ----------

values, base = np.histogram(dist_df, bins=40)
cumulative = np.cumsum(values)
plt.plot(base[:-1], cumulative, c='blue')
plt.show()

# COMMAND ----------

import seaborn as sns

sns.ecdfplot(dist_df)

# COMMAND ----------

dist_df['MEMBER_ID'] = df_final['MEMBER_ID']
dist_df.to_csv(folder_path + "ikea_dist_matrix_part5_0620.csv")

# COMMAND ----------

dist_df = pd.read_csv(folder_path + "ikea_dist_matrix_part5_0620.csv", index_col=0)
dist_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Getting Results
# MAGIC - Compare the list of member's distance matrix in the testing set (around 250K), with the list of member actually made their first purchase (around 27K)
# MAGIC - Find intersection between two list (and find accuracy)
# MAGIC - Also find out the precision / recall curve to determine the threshold

# COMMAND ----------

converted_member_list = pd.read_csv("/dbfs/mnt/rex/ikea_prospect/0609_data/ikea_first_purchase_limited_member_list_0609.csv", index_col=None)
converted_member_list

# COMMAND ----------

dist_df = pd.read_csv(folder_path + "ikea_dist_matrix_part1_0620.csv", index_col=0)
# for i in range(25):
#     tmp_df = pd.read_csv(folder_path + "ikea_dist_matrix_part{}_0614.csv".format(i+1), index_col=0)
#     dist_df = pd.concat([dist_df, tmp_df], ignore_index=True)

dist_df = dist_df.rename(columns={'0': 'distance'})
dist_df = dist_df[['MEMBER_ID', 'distance']]
dist_df

# COMMAND ----------

sorted_dist_df = dist_df.sort_values(by='distance')
sorted_dist_df

# COMMAND ----------

sorted_dist_df['distance'].describe()

# COMMAND ----------

matched_df = sorted_dist_df.merge(converted_member_list, on='MEMBER_ID', how='inner')
matched_df

# COMMAND ----------

correct_result_df = sorted_dist_df[sorted_dist_df['distance'] < 100].merge(converted_member_list, on='MEMBER_ID', how='inner')
correct_result_df

# COMMAND ----------

tp = correct_result_df.shape[0]
all_pos_label = sorted_dist_df[sorted_dist_df['distance'] < 100].shape[0]
actual_pos_label = matched_df.shape[0]
print(tp)
print(all_pos_label)
print(actual_pos_label)

# COMMAND ----------

precision = tp / all_pos_label
precision

# COMMAND ----------

recall = tp / actual_pos_label
recall

# COMMAND ----------

perc = np.linspace(0, 1, 11)
perc_df = sorted_dist_df['distance'].describe(percentiles=perc)
perc_df

# COMMAND ----------

precision_list = []
recall_list = []
for thres in perc_df.iloc[4:-1]:
    all_pos_label = sorted_dist_df[sorted_dist_df['distance'] <= thres].shape[0]
    correct_result_df = sorted_dist_df[sorted_dist_df['distance'] < thres].merge(converted_member_list, on='MEMBER_ID', how='inner')
    tp = correct_result_df.shape[0]
    precision = tp / all_pos_label
    recall = tp / actual_pos_label
    precision_list.append(precision)
    recall_list.append(recall)
#     print(precision, recall)
df_dict = {'precision': precision_list, 'recall': recall_list}
pr_df = pd.DataFrame.from_dict(df_dict)
pr_df

# COMMAND ----------

pr_df.plot.line(x='recall', y='precision', style='-o', ylim=(0,0.02))

# COMMAND ----------

