# %% Import package to use
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
import os
# %% Load dataset
DATA_PATH = "/home/lkh256/Studio/Asthma/ISE_FeNO_eos/data"
# df_init = pd.read_excel(os.path.join(DATA_PATH, "asthma_dataset_final.xlsx"), \
#                         sheet_name="746 (소아2 중복3 missing 8제거)")

df_init = pd.read_excel(os.path.join(DATA_PATH, "20210913 ISE-FENO eos.xlsx"), \
                        sheet_name="669 (77 missing eos제외)")

print("Number of samples = {}".format(len(df_init)))
display(df_init.head())

# %%
from tableone import TableOne

columns = ['AgeAtDx', 'Sex_f1m2', 'BMI', 'Smk', 'Sx_dyspnea', 
           'Sx_cough', 'Sx_wheezing', 'Co_rhinitis', 'Skintest', 
           'IgE', 'Lab_EosCount', 'FeNO', 
           'ISE_Eos', 'ISE_Neu']
categorical = ['Sex_f1m2', 'Smk', 'Sx_dyspnea', 'Sx_cough', 'Sx_wheezing', 'Skintest', 'Co_rhinitis']
nonnormal = ['AgeAtDx', 'IgE', 'Lab_EosCount', 'FeNO', 'ISE_Eos', 'ISE_Neu']
groupby = ['Asthma']

table = TableOne(df_init, columns=columns, 
                 categorical=categorical, 
                 nonnormal=nonnormal, 
                 groupby=groupby, decimals=2, pval=True)
display(table)
table.to_excel('mytable.xlsx')
# %%
import seaborn as sns
for var in list(set(columns) - set(categorical)):
    plt.title(var)
    sns.distplot(x=df_init[var])
    plt.show()
# %%
