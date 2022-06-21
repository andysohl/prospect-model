# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Data Extraction V3
# MAGIC This version, we'll build the member's profile by each of the category in the banners, instead of the whole banner activity
# MAGIC Only select
# MAGIC 
# MAGIC #### Data to be used in building profile
# MAGIC Sales and frequencies by category if possible (This version we focus on retail banners):
# MAGIC 1. WEHK (Need to split WCL & USS?)
# MAGIC 2. SEHK
# MAGIC 3. MNHK

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

# COMMAND ----------

# Connect with the database
password=dbutils.secrets.get(scope = "DATALAKE-PD", key = "synapse-pd01-dfsg-datafactory-pwd")
url = "jdbc:sqlserver://sql-gpdl01-pd-sea-df-pd-sql01.database.windows.net:1433;database=sqldwh-gpdl01-pd-sea-dl01-sqldwh01;user=dfsg_datafactory@sql-gpdl01-pd-sea-df-pd-sql01;password="+password+";encrypt=true;trustServerCertificate=true;"

# COMMAND ----------

folder_path = "/dbfs/mnt/rex/ikea_prospect/"

# COMMAND ----------

# MAGIC %md
# MAGIC #### Find the member's activity one month before making the first purchase at IKEA

# COMMAND ----------

# MAGIC %md
# MAGIC ##### WEHK

# COMMAND ----------

train_start_date = '2020-07-30'
train_end_date = '2022-02-28'

# COMMAND ----------

df_wehk = pd.read_csv(folder_path + "0609_data/ikea_first_purchase_member_profile_WEHK_0609.csv", index_col=None)
df_wehk

# COMMAND ----------

df_wehk_pivot = df_wehk.pivot(index='MEMBER_ID', columns='department_name', values=['wehk_freq', 'wehk_sales']).reset_index()
df_wehk_pivot.columns = [f'{i}_{j}' for i, j in df_wehk_pivot.columns]
df_wehk_pivot

# COMMAND ----------

# MAGIC %md
# MAGIC ##### SEHK

# COMMAND ----------

df_sehk = pd.read_csv(folder_path + "0609_data/ikea_first_purchase_member_profile_SEHK_0609.csv", index_col=None)
df_sehk

# COMMAND ----------

df_sehk_pivot= df_sehk.pivot(index='MEMBER_ID', columns='category_name', values=['sehk_freq', 'sehk_sales']).reset_index()
df_sehk_pivot.columns = [f'{i}_{j}' for i, j in df_sehk_pivot.columns]
df_sehk_pivot

# COMMAND ----------

# MAGIC %md
# MAGIC ##### MNHK

# COMMAND ----------

df_mnhk = pd.read_csv(folder_path + "0609_data/ikea_first_purchase_member_profile_MNHK_0609.csv", index_col=None)
df_mnhk

# COMMAND ----------

df_mnhk_pivot= df_mnhk.pivot(index='MEMBER_ID', columns='bunit_name', values=['mnhk_freq', 'mnhk_sales']).reset_index()
df_mnhk_pivot.columns = [f'{i}_{j}' for i, j in df_mnhk_pivot.columns]
df_mnhk_pivot

# COMMAND ----------

# Combining data from all banners
final_df = pd.merge(df_wehk_pivot, df_sehk_pivot, how='outer')
final_df = pd.merge(final_df, df_mnhk_pivot, how='outer')
final_df

# COMMAND ----------

final_df = final_df.dropna(axis=1, how='all')
final_df

# COMMAND ----------

final_df = final_df.fillna(0)
final_df

# COMMAND ----------

final_df.to_csv("/dbfs/mnt/rex/ikea_prospect_v2/ikea_full_member_profile_retail_only_0620.csv", index=False)

# COMMAND ----------

final_df = pd.read_csv("/dbfs/mnt/rex/ikea_prospect_v2/ikea_full_member_profile_retail_only_0620.csv", index_col=None)
final_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit all member profiles in a nearest neighbor graph

# COMMAND ----------

# Fit the whole dataset into a KNN neighbor model
knn = NearestNeighbors()
knn.fit(final_df.iloc[:, 1:])

# COMMAND ----------

# Save model as pickle
knnPickle = open('/dbfs/mnt/rex/ikea_prospect_v2/knn_model_retail_0620.pkl', 'wb')

# source, destination 
pickle.dump(knn, knnPickle)
knnPickle.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### LOAD MODEL

# COMMAND ----------

# Load the model
knn = pickle.load(open('/dbfs/mnt/rex/ikea_prospect_v2/knn_model_retail_0620.pkl', 'rb'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get any member's profile and find the distance between the member and the dataset

# COMMAND ----------

column_list = final_df.columns.tolist()
column_list = pd.Series(column_list)
column_list.to_csv("/dbfs/mnt/rex/ikea_prospect_v2/final_df_column_list_0620.csv", index=False)

# COMMAND ----------

column_list = pd.read_csv("/dbfs/mnt/rex/ikea_prospect_v2/final_df_column_list_0620.csv", index_col=None)
column_list.iloc[0,0] = 'MEMBER_ID'
column_list

# COMMAND ----------

test_final_df = pd.DataFrame(columns=column_list['0'].tolist())
test_final_df

# COMMAND ----------

# Send the query to data lake, retrieve the result, and save it as dataframe, then CSV file
end_date = datetime.today() - relativedelta(days=1)
start_date = end_date - relativedelta(months=1)
end_date = end_date.strftime("%Y-%m-%d")
start_date = start_date.strftime("%Y-%m-%d")
member_id = str(934480004873466)

print(start_date)
print(end_date)
print(member_id)

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
and a.[TIME_PERIOD] BETWEEN '{}' AND '{}'
and a.member_id = '{}'
GROUP BY a.member_id, c.department_no, c.department_name
"""

# COMMAND ----------

df_spark = (spark.read
          .format("jdbc")
          .option("url", url)
          .option("query", test_wehk_sql.format(start_date, end_date, member_id))
          .load()
            )
test_wehk_df = df_spark.toPandas()
test_wehk_df['MEMBER_ID'] = pd.to_numeric(test_wehk_df['MEMBER_ID'])
test_wehk_df

# COMMAND ----------

if test_wehk_df.empty:
    wehk_columns = ['MEMBER_ID'] + [x for x in column_list['0'].tolist() if x.startswith("wehk_")]
    test_wehk_df_pivot = pd.DataFrame(columns=wehk_columns)
    row = []
    for i in range(len(wehk_columns)):
        if i == 0:
           row.append(member_id)
        else:
           row.append(0)
    test_wehk_df_pivot.loc[0] = row
else:
    test_wehk_df_pivot = test_wehk_df.pivot(index='MEMBER_ID', columns='department_name', values=['wehk_freq', 'wehk_sales']).reset_index()
    test_wehk_df_pivot.columns = [f'{i}_{j}' if j != '' else f'{i}' for i, j in test_wehk_df_pivot.columns]

test_wehk_df_pivot

# COMMAND ----------

row_dict = {}
for column in test_wehk_df_pivot.columns:
    row_dict[column] = test_wehk_df_pivot.loc[0, column]
test_final_df = test_final_df.append(row_dict, ignore_index=True)
test_final_df

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
and a.[TIME_PERIOD] BETWEEN '{}' AND '{}'
and a.member_id = '{}'
GROUP BY a.member_id, c.category_name
"""

# COMMAND ----------

df_spark = (spark.read
          .format("jdbc")
          .option("url", url)
          .option("query", test_sehk_sql.format(start_date, end_date, member_id))
          .load()
            )
test_sehk_df = df_spark.toPandas()
test_sehk_df['MEMBER_ID'] = pd.to_numeric(test_sehk_df['MEMBER_ID'])
test_sehk_df

# COMMAND ----------

if test_sehk_df.empty:
    sehk_columns = ['MEMBER_ID'] + [x for x in column_list['0'].tolist() if x.startswith("sehk_")]
    test_sehk_df_pivot = pd.DataFrame(columns=sehk_columns)
    row = []
    for i in range(len(sehk_columns)):
        if i == 0:
           row.append(member_id)
        else:
           row.append(0)
    test_sehk_df_pivot.loc[0] = row
else:
    test_sehk_df_pivot = test_sehk_df.pivot(index='MEMBER_ID', columns='category_name', values=['sehk_freq', 'sehk_sales']).reset_index()
    test_sehk_df_pivot.columns = [f'{i}_{j}' if j != '' else f'{i}' for i, j in test_sehk_df_pivot.columns]

test_sehk_df_pivot

# COMMAND ----------

# row_dict = {}
for column in test_sehk_df_pivot.columns:
    test_final_df.loc[0, column] = test_sehk_df_pivot.loc[0, column]
# test_final_df = test_final_df.append(row_dict, ignore_index=True)
test_final_df

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
and a.[TIME_PERIOD] BETWEEN '{}' AND '{}'
and a.member_id = '{}'
GROUP BY a.member_id, c.bunit_name
"""

# COMMAND ----------

df_spark = (spark.read
          .format("jdbc")
          .option("url", url)
          .option("query", test_mnhk_sql.format(start_date, end_date, member_id))
          .load()
            )
test_mnhk_df = df_spark.toPandas()
test_mnhk_df['MEMBER_ID'] = pd.to_numeric(test_mnhk_df['MEMBER_ID'])
test_mnhk_df

# COMMAND ----------

if test_mnhk_df.empty:
    mnhk_columns = ['MEMBER_ID'] + [x for x in column_list['0'].tolist() if x.startswith("mnhk_")]
    test_mnhk_df_pivot = pd.DataFrame(columns=mnhk_columns)
    row = []
    for i in range(len(mnhk_columns)):
        if i == 0:
           row.append(member_id)
        else:
           row.append(0)
    test_mnhk_df_pivot.loc[0] = row
else:
    test_mnhk_df_pivot = test_mnhk_df.pivot(index='MEMBER_ID', columns='bunit_name', values=['mnhk_freq', 'mnhk_sales']).reset_index()
    test_mnhk_df_pivot.columns = [f'{i}_{j}' if j != '' else f'{i}' for i, j in test_mnhk_df_pivot.columns]

test_mnhk_df_pivot

# COMMAND ----------

# row_dict = {}
for column in test_mnhk_df_pivot.columns:
    test_final_df.loc[0, column] = test_mnhk_df_pivot.loc[0, column]
# test_final_df = test_final_df.append(row_dict, ignore_index=True)
test_final_df

# COMMAND ----------

test_final_df.shape

# COMMAND ----------

# test_final_df = test_final_df.dropna(axis=1, how='all')
test_final_df = test_final_df.fillna(0)
test_final_df

# COMMAND ----------

test_final_df = test_final_df.apply(pd.to_numeric, errors='ignore')
print(test_final_df.shape)

# COMMAND ----------

test_array = test_final_df.iloc[0,1:].tolist()
test_array

# COMMAND ----------

dist, ind = knn.kneighbors([test_array], 10, return_distance=True)

# COMMAND ----------

dist

# COMMAND ----------

ind

# COMMAND ----------

print(final_df.iloc[ind[0][0],1:].tolist())
print(test_final_df.iloc[0,1:].tolist())

# COMMAND ----------

np.mean(dist)

# COMMAND ----------

