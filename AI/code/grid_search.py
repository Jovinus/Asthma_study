# %% Import package to use
import re
import pandas as pd
import xgboost as xgb
from IPython.display import display
from my_module import *
import os
import argparse
pd.set_option("display.max_columns", None)

# %% Argparser
parser = argparse.ArgumentParser(description='Feature Combinations')
parser.add_argument('--feature', type=str, nargs='+', default=list())
parser.add_argument('--file_nm', type=str)
parser.add_argument('--text_mode', type=bool, default=False)
parser.add_argument('--gpu_id', type=int, default=0)
args = vars(parser.parse_args())

# %% Load dataset
DATA_PATH = "/home/lkh256/Studio/Asthma/DB/data"
df_init = pd.read_csv(os.path.join(DATA_PATH, 'asthma_ai_dataset.csv'), encoding='utf-8')
df_init['IndexDate'] = df_init['IndexDate'].astype('datetime64')

print("Number of samples = {}".format(len(df_init)))
display(df_init.head())

# %% Check missings
print("Check their is any missing variables in dataset: \n", df_init.isnull().sum())

# %%
baseline = ['MBPT_result', 'maxFall_FEV1_p', 'PC20_32']

baseline_txt = [i for i in df_init.columns if "baseline" in i]
salline_txt = [i for i in df_init.columns if "saline" in i]

subs = r"0.05|0.5|2.0|8.0|16.0|32.0"
mbpt_txt = [x for x in df_init.columns if re.search(subs, str(x))]

if args['text_mode'] == True:
    feature_mask = args['feature'] + mbpt_txt
else:
    feature_mask = args['feature']

# %%
train_set, test_set = df_init.query('IndexDate.dt.year <= 2018', engine='python'), df_init.query('IndexDate.dt.year > 2018', engine='python')

print("Train set size = {}".format(len(train_set)))
print("Test set size = {}".format(len(test_set)))
# %%
from sklearn.model_selection import ParameterGrid
param_grids = {'objective':['binary:logistic'], 
              'eval_metric':['logloss'], 
              'tree_method':['gpu_hist'], 
              'gpu_id':[args['gpu_id']], 
              'learning_rate':[0.01, 0.001], 
              'max_depth':[1, 2, 3, 4, 5],
              'lambda':[1, 2, 3],
              'gamma':[0, 0.1, 0.2, 0.3]
              }   

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import log_loss
from tqdm import tqdm

result = pd.DataFrame()
fig_result = pd.DataFrame()

for param_grid in tqdm(ParameterGrid(param_grids)):
    
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=6)
    val_specificity = list()
    val_sensitivity = list()
    val_ppv = list()
    val_npv = list()
    val_f1 = list()
    val_accuracy = list()
    val_threshold_of_interest = list()
    val_roc_auc = list()
    val_pr_auc = list()
    val_loss = list()
    
    test_specificity = list()
    test_sensitivity = list()
    test_ppv = list()
    test_npv = list()
    test_f1 = list()
    test_accuracy = list()
    test_threshold_of_interest = list()
    test_roc_auc = list()
    test_pr_auc = list()
    test_loss = list()
    
    fig_fpr = list()
    fig_tpr = list()
    fig_precision = list()
    fig_recall = list()
    
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)
    
    for train_index, validation_index in skf.split(train_set, train_set['Asthma']):
        X_train = train_set.iloc[train_index][feature_mask].values
        y_train = train_set.iloc[train_index]['Asthma'].values
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        X_validation = train_set.iloc[validation_index][feature_mask].values
        y_validation = train_set.iloc[validation_index]['Asthma'].values
        dvalidation = xgb.DMatrix(X_validation, label=y_validation)
        
        X_test = test_set[feature_mask].values
        y_test = test_set['Asthma'].values
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        model_xgb = xgb.train(param_grid, dtrain, 
                              num_boost_round=20000, 
                              evals=[(dvalidation, 'validation')], 
                              verbose_eval=0, 
                              early_stopping_rounds=1000)
        
        specificity, sensitivity, ppv, npv, f1, accuracy, threshold_of_interest, roc_auc, pr_auc, fpr, tpr, precision, recall = performances_hard_decision(y_validation, model_xgb.predict(dvalidation), youden=True)
        loss = log_loss(y_true=y_validation, y_pred=model_xgb.predict(dvalidation))
        
        val_specificity.append(specificity)
        val_sensitivity.append(sensitivity)
        val_ppv.append(ppv)
        val_npv.append(npv)
        val_f1.append(f1)
        val_accuracy.append(accuracy)
        val_threshold_of_interest.append(threshold_of_interest)
        val_roc_auc.append(roc_auc)
        val_pr_auc.append(pr_auc)
        val_loss.append(loss)
        
        specificity, sensitivity, ppv, npv, f1, accuracy, threshold_of_interest, roc_auc, pr_auc, fpr, tpr, precision, recall = performances_hard_decision(y_test, model_xgb.predict(dtest), youden=True)
        loss = log_loss(y_true=y_test, y_pred=model_xgb.predict(dtest))
        
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        
        interp_precision = np.interp(mean_recall, np.flip(recall), np.flip(precision))
        
        test_specificity.append(specificity)
        test_sensitivity.append(sensitivity)
        test_ppv.append(ppv)
        test_npv.append(npv)
        test_f1.append(f1)
        test_accuracy.append(accuracy)
        test_threshold_of_interest.append(threshold_of_interest)
        test_roc_auc.append(roc_auc)
        test_pr_auc.append(pr_auc)
        test_loss.append(loss)
        
        fig_fpr.extend(mean_fpr.tolist())
        fig_tpr.extend(interp_tpr.tolist())
        fig_precision.extend(interp_precision.tolist())
        fig_recall.extend(mean_recall.tolist())
    
    metric = pd.DataFrame({'val_specificity':val_specificity, 
                           'val_sensitivity':val_sensitivity,
                           'val_ppv':val_ppv,
                           'val_npv':val_npv,
                           'val_f1':val_f1, 
                           'val_accuracy':val_accuracy,
                           'val_threshold':val_threshold_of_interest, 
                           'val_roc_auc':val_roc_auc, 
                           'val_pr_auc':val_pr_auc,
                           'val_loss':val_loss, 
                           'test_specificity':test_specificity, 
                           'test_sensitivity':test_sensitivity,
                           'test_ppv':test_ppv,
                           'test_npv':test_npv,
                           'test_f1':test_f1, 
                           'test_accuracy':test_accuracy,
                           'test_threshold':test_threshold_of_interest, 
                           'test_roc_auc':test_roc_auc, 
                           'test_pr_auc':test_pr_auc,
                           'test_loss':test_loss
                            })
    metric['gamma'] = param_grid['gamma']
    metric['lambda'] = param_grid['lambda']
    metric['learning_rate'] = param_grid['learning_rate']
    metric['max_depth'] = param_grid['max_depth']
    
    result = pd.concat((result, metric), axis=0)
    
    fig_roc_prc = pd.DataFrame({'fpr':fig_fpr, 'tpr':fig_tpr, 'precision':fig_precision, 'recall':fig_recall})
    fig_roc_prc['gamma'] = param_grid['gamma']
    fig_roc_prc['lambda'] = param_grid['lambda']
    fig_roc_prc['learning_rate'] = param_grid['learning_rate']
    fig_roc_prc['max_depth'] = param_grid['max_depth']
    
    fig_result = pd.concat((fig_result, fig_roc_prc), axis=0)

# %%
result.to_csv("../../result/ai_metric_results_" + args['file_nm'] + ".csv", index=False)
fig_result.to_csv("../../result/ai_fig_results_" + args['file_nm'] + ".csv", index=False)

# %%
specificity, sensitivity, ppv, npv, f1, accuracy, threshold_of_interest, roc_auc, pr_auc, fpr, tpr, precision, recall = performances_hard_decision(test_set['Asthma'], test_set['MBPT_result'])
# %%
baseline = pd.DataFrame({'test_specificity':[specificity], 'test_sensitivity':[sensitivity], 
                         'test_ppv':[ppv], 'test_npv':[npv], 'test_f1':[f1], 'test_accuracy':[accuracy], 
                         'test_threshold':[threshold_of_interest], 
                         'test_roc_auc':[roc_auc], 'test_pr_auc':[pr_auc], 'test_loss':[0], 'feature':['baseline']})

baseline.to_csv("../result/baseline_metric.csv", index=False)
# %%
