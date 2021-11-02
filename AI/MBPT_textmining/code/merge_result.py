# %%
import pandas as pd
import datatable
import numpy as np
import re
from IPython.display import display
pd.set_option("display.max_columns", None)
# %%
df_db = pd.read_excel("../data/dataset_20210806.xlsx")
df_db['date'] = df_db['date'].astype('datetime64')
df_mbpt = pd.read_csv("../data/AI_asthmaDX_parsed.csv")
df_mbpt['INDEX_DATE'] = df_mbpt['INDEX_DATE'].astype('datetime64')
# %%
df_result = pd.merge(df_db, df_mbpt, 
                     how='left', left_on=['ID', 'date'], 
                     right_on=['ID', 'INDEX_DATE'])

df_result.to_csv("../../data/asthma_dataset.csv", index=False, encoding='utf-8-sig')
df_result.to_excel("../../data/asthma_dataset.xlsx", index=False)
# %%
