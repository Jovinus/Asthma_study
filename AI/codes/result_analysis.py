# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import re
from my_module import *
pd.set_option("display.max_columns", None)

# %%

fig_results = pd.DataFrame()
metric_results = pd.DataFrame()

fig_data_list = glob("../result/ai_fig*csv")
metric_data_list = glob("../result/ai_metric*csv")

fig_data_list.sort()
metric_data_list.sort()

for metric_data, fig_data in zip(metric_data_list, fig_data_list):
    df_metric = pd.read_csv(metric_data)
    df_metric = df_metric.assign(mean_val_loss = lambda x: x.groupby(['gamma', 'lambda', 'learning_rate', 'max_depth'])['val_loss'].transform(lambda x: x.mean()))
    df_metric['feature'] = metric_data[28:-4]
    df_metric = df_metric.query("mean_val_loss == mean_val_loss.min()")
    
    hyper_gamma = df_metric['gamma'].head(1).values[0]
    hyper_lambda = df_metric['lambda'].head(1).values[0]
    hyper_learning_rate = df_metric['learning_rate'].head(1).values[0]
    hyper_max_depth = df_metric['max_depth'].head(1).values[0]
    
    df_fig = pd.read_csv(fig_data)
    df_fig = df_fig.rename(columns={'lambda':'hyper_lambda'})
    df_fig['feature'] = metric_data[28:-4]
    df_fig = df_fig.query("(gamma == @hyper_gamma) & (hyper_lambda == @hyper_lambda) & (learning_rate == @hyper_learning_rate) & (max_depth == @hyper_max_depth)")
    
    metric_results = pd.concat((metric_results, df_metric), axis=0)
    fig_results = pd.concat((fig_results, df_fig), axis=0)

# %%
feature_order = ['mbpt_result_pc20', 'mbpt_result_feno', 'mbpt_result_ise', 'mbpt_result_feno_ise',
                 'mbpt_txt', 'mbpt_txt_feno', 'mbpt_txt_ise', 'mbpt_txt_feno_ise']

# %%
metric_columns = [x for x in metric_results.columns if re.search('test', x)]

for metric in metric_columns:
    fig, ax = plt.subplots(1, 1, figsize=(20,10))
    sns.boxplot(x='feature', y=metric, data=metric_results, ax=ax, order=feature_order)
    plt.grid()
    plt.xticks(rotation=45)
    plt.show()
    
# %%
for feature in feature_order:
    plot_roc_curve(x='fpr', y='tpr', data=fig_results.query("feature == @feature"), mean=metric_results.query("feature == @feature")['test_roc_auc'].mean(), std=metric_results.query("feature == @feature")['test_roc_auc'].std())
    plot_prc_curve(x='recall', y='precision', data=fig_results.query("feature == @feature"), mean=metric_results.query("feature == @feature")['test_pr_auc'].mean(), std=metric_results.query("feature == @feature")['test_pr_auc'].std())
# %%
metric_results.to_csv('../result/selected_metric/selected_metric.csv', index=False, encoding='utf-8')
fig_results.to_csv('../result/selected_metric/selected_fig.csv', index=False, encoding='utf-8')
# %%
