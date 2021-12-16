# %%
import pandas as pd
from glob import glob
import re
import numpy as np
pd.set_option("display.max_columns", None)
# %%
DATAPATH = "../data/Skin_Test/"
list_data = glob(DATAPATH + '*skin*')
list_data.sort()

df_id = pd.read_excel("../data/Skin_Test/asthma_patient_list.xlsx")
# %%
df_skin = pd.DataFrame()

for file_nm in list_data:
    tmp_df = pd.read_excel(file_nm)
    df_skin = pd.concat((df_skin, tmp_df), axis=0)
# %%
df_skin = df_skin.reset_index(drop=True)
df_skin['검사명#8'].value_counts()
# %%
EXTRACTPATTERN = r'(?P<grade>\d+)[(]\d+[*]\d+[)]'

def extract_pattern(x, pattern):
    try:
        return re.search(pattern, x).group('grade')
    except:
        return np.nan
# %%

df_skin = df_skin.assign(test_result = lambda x: x['검사결과수치값#11'].apply(lambda x: extract_pattern(x, EXTRACTPATTERN)).astype(float))
# %%
df_skin = df_skin[~df_skin['검사명#8'].str.contains('Histamine')].groupby(['환자번호#1', 'patient_indi#13', '처방일자#5'])['test_result']\
                .apply(lambda x: (x >= 3).any()).reset_index(drop=False)
# %%
df_skin = pd.merge(df_skin, df_id, 
                   how='left', 
                   left_on=['patient_indi#13'], 
                   right_on=['patient_indi']).drop(columns=['환자번호#1', 'patient_indi#13', 'patient_indi'])

df_skin = df_skin[['PtID', '처방일자#5', 'test_result']].drop_duplicates()
# %%
df_skin.to_csv("../data/Skin_Test/skin_test.csv", index=False)
# %%
