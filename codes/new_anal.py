# %%
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_columns", None)

# %%
df_orig = pd.read_excel('../data/추가결과_이진영_20180907.xlsx', sheet_name='db')
display(df_orig.head())

# %%
df_set = df_orig.query('Bl_eos_count.notnull()', engine='python')
# %%
print(df_set.query('Bl_eos_count < 100')['sp_cut'].value_counts(normalize=True))
print(df_set.query('Bl_eos_count < 200')['sp_cut'].value_counts(normalize=True))
print(df_set.query('Bl_eos_count < 300')['sp_cut'].value_counts(normalize=True))
print(df_set.query('Bl_eos_count < 400')['sp_cut'].value_counts(normalize=True))
print(df_set.query('Bl_eos_count < 500')['sp_cut'].value_counts(normalize=True))
print(df_set.query('Bl_eos_count > 500')['sp_cut'].value_counts(normalize=True))

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

df_tmp = pd.DataFrame([])
for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
    tmp_df = df_set.query('Bl_eos_count < ' + str(100 * (i + 1)) + ' & ' + 'Bl_eos_count >= ' + str(100 * i))['asthma'].value_counts(normalize=True).reset_index()
    tmp_df['feature'] = 'Bl_eos_count < ' + str(100 * (i + 1)) + ' & ' + 'Bl_eos_count >= ' + str(100 * i)
    df_tmp = pd.concat([df_tmp, tmp_df], axis=0)
    
sns.barplot(x='feature', y='asthma', hue='index', data=df_tmp)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

df_tmp = pd.DataFrame([])
for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    tmp_df = df_set.query('FeNO < ' + str(10 * (i + 1)) + ' & ' + 'FeNO >= ' + str(10 * i))['sp_cut'].value_counts(normalize=True).reset_index()
    tmp_df['feature'] = 'FeNO < ' + str(10 * (i + 1)) + ' & ' + 'FeNO >= ' + str(10 * i)
    df_tmp = pd.concat([df_tmp, tmp_df], axis=0)
    
sns.barplot(x='feature', y='sp_cut', hue='index', data=df_tmp)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

df_tmp = pd.DataFrame([])
for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    tmp_df = df_set.query('Bl_eos_count < ' + str(10 * (i + 1)) + ' & ' + 'Bl_eos_count >= ' + str(10 * i))['sp_cut'].value_counts(normalize=True).reset_index()
    tmp_df['feature'] = 'Bl_eos_count < ' + str(10 * (i + 1)) + ' & ' + 'Bl_eos_count >= ' + str(10 * i)
    df_tmp = pd.concat([df_tmp, tmp_df], axis=0)
    
sns.barplot(x='feature', y='sp_cut', hue='index', data=df_tmp)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

df_tmp = pd.DataFrame([])
for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    tmp_df = df_set.query('FeNO < ' + str(10 * (i + 1)) + ' & ' + 'FeNO >= ' + str(10 * i))['asthma'].value_counts(normalize=True).reset_index()
    tmp_df['feature'] = 'FeNO < ' + str(10 * (i + 1)) + ' & ' + 'FeNO >= ' + str(10 * i)
    df_tmp = pd.concat([df_tmp, tmp_df], axis=0)
    
sns.barplot(x='feature', y='asthma', hue='index', data=df_tmp)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()
# %%
