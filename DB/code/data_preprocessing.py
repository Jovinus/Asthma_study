# %% Import package to use
import pandas as pd
import numpy as np
import re
pd.set_option("display.max_columns", None)
# %%
df_cur_db = pd.read_excel("../data/asthma_dataset_final.xlsx", sheet_name='746 (소아2 중복3 missing 8제거)')
# %%
df_new_db = pd.read_excel("../data/validation_set_parsed.xlsx")

# %%
subs = r"0.05|0.5|2.0|8.0|16.0|32.0"
mbpt_txt = [x for x in df_cur_db.columns if re.search(subs, str(x))]
baseline_feature = ['PtID', 'IndexDate', 'AgeAtDx', 'Sex_f1m2', 'BMI', 'Smk', 'Sx_dyspnea', 'Sx_cough', 'Sx_wheezing', 'Co_rhinitis', 'Skintest', 'IgE', 'Sx_cough', 'Sx_wheezing', 'Asthma', 'MBPT_result', 'PC20_16', 'FeNO', 'ISE_Eo3%', 'ISE_Eos', 'ISE_Neu']
# %%
df_new_db = df_new_db[baseline_feature + mbpt_txt]
df_new_db = df_new_db.assign(Sex_f1m2=lambda x: x['Sex_f1m2'].map({'F':1, 'M':2}), 
                             Smk=lambda x: x['Smk'].map({'Never-smoker':0, 'Never smoker':0, 'Non-smoker':0, 'Ex-smoker':1, '과거흡연, 현재금연':1}).fillna(0), 
                             Skintest=lambda x: x['Skintest'].map({'P':2, 'B':1, 'N':0, 'N(Dermo)':0})
                             )
df_new_db = df_new_db.query('ISE_Eos.notnull()', engine='python')
# %%
df_cur_db = df_cur_db[baseline_feature + mbpt_txt]
df_cur_db = df_cur_db.assign(Skintest=lambda x: x['Skintest'].map({'P':2, 'B':1, 'N':0, 'N(Dermo)':0}))
# %%
df_db = pd.concat((df_cur_db, df_new_db), axis=0)
# %%
df_db.to_csv('../data/asthma_ai_dataset.csv', index=False, encoding='utf-8')

# %%
