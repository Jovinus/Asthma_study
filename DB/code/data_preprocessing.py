# %% Import package to use
import pandas as pd
import numpy as np
import re
pd.set_option("display.max_columns", None)
# %%
df_cur_ai_db = pd.read_excel("../data/asthma_dataset_final.xlsx", sheet_name='746 (소아2 중복3 missing 8제거)')

df_cur_cut_db = pd.read_excel('../data/20210913 ISE-FENO eos.xlsx', sheet_name='669 (77 missing eos제외)')
# %%
df_new_db = pd.read_excel("../data/validation_set_parsed.xlsx")

# %%
subs = r"0.05|0.5|2.0|8.0|16.0|32.0"
mbpt_txt = [x for x in df_cur_ai_db.columns if re.search(subs, str(x))]
baseline_feature = ['PtID', 'IndexDate', 'AgeAtDx', 'Sex_f1m2', 'BMI', 
                    'Smk', 'Sx_dyspnea', 'Sx_cough', 'Sx_wheezing', 
                    'Co_rhinitis', 'Skintest', 'IgE', 'Asthma', 'MBPT_result', 'PC20_16', 
                    'FeNO', 'ISE_Eo3%', 'ISE_Eos', 'ISE_Neu', 'Lab_EosCount', 'Lab_Eos(%)']
# %%
df_new_db = df_new_db[baseline_feature + mbpt_txt]
df_new_db = df_new_db.assign(Sex_f1m2=lambda x: x['Sex_f1m2'].map({'F':1, 'M':2}), 
                             Smk=lambda x: x['Smk'].map({'Never-smoker':0, 'Never smoker':0, 'Non-smoker':0, 'Ex-smoker':1, '과거흡연, 현재금연':1}), 
                             Skintest=lambda x: x['Skintest'].map({'P':2, 'B':1, 'N':0, 'N(Dermo)':0})
                             )
df_new_db = df_new_db.query('ISE_Eos.notnull()', engine='python')
# %%
df_cur_ai_db = df_cur_ai_db[baseline_feature + mbpt_txt]
df_cur_ai_db = df_cur_ai_db.assign(Skintest=lambda x: x['Skintest'].map({'P':2, 'B':1, 'N':0, 'N(Dermo)':0}), 
                                   Smk=lambda x: x['Smk'].map({0:0, 1:1, 2:1}))

df_cur_cut_db = df_cur_cut_db[baseline_feature + mbpt_txt]
# %%
df_ai_db = pd.concat((df_cur_ai_db, df_new_db), axis=0)

df_cut_db = pd.concat((df_cur_cut_db, df_new_db), axis=0)

# %%
df_skin = pd.read_csv("../data/Skin_Test/skin_test.csv")
df_skin = df_skin.assign(IndexDate = lambda x: pd.to_datetime(x['처방일자#5']), 
                         Skintest_new = lambda x: x['test_result'].astype(int)).drop(columns=['처방일자#5', 'test_result'])

df_ai_db = pd.merge(df_ai_db, df_skin, on=['PtID', 'IndexDate'], how='left')
df_ai_db = df_ai_db.assign(SkinTest = lambda x: np.where(x['Skintest_new'].notnull(), x['Skintest_new'], x['Skintest'].map({1:1, 2:1})))

df_cut_db = pd.merge(df_cut_db, df_skin, on=['PtID', 'IndexDate'], how='left')
df_cut_db = df_cut_db.assign(SkinTest = lambda x: np.where(x['Skintest_new'].notnull(), x['Skintest_new'], x['Skintest'].map({1:1, 2:1})))
# %%
df_ai_db.to_csv('../data/asthma_ai_dataset.csv', index=False, encoding='utf-8')
df_cut_db.to_csv('../data/ISE_3_cut_dataset.csv', index=False, encoding='utf-8')

# %%
