# %% Import package to use
import numpy as np
from numpy.lib.function_base import disp
import re
import pandas as pd
import xgboost as xgb
from IPython.display import display
pd.set_option("display.max_columns", None)
import os

# %% Load dataset
DATA_PATH = "/home/lkh256/Studio/Asthma/AI/data"
df_init = pd.read_excel(os.path.join(DATA_PATH, "asthma_dataset_final.xlsx"), \
                        sheet_name="746 (소아2 중복3 missing 8제거)")

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

# %%
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df_init,  
                                       random_state=1004, 
                                       stratify=df_init['Asthma'], 
                                       test_size=0.2)

print("Train set size = {}".format(len(train_set)))
print("Test set size = {}".format(len(test_set)))
# %%
feature_mask = baseline_txt + mbpt_txt + baseline

# %% Cross-Val Model

from tqdm import tqdm
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score
import numpy as np
import pickle

hyper_param_depth = [1, 2, 3, 4, 5]
hyper_param_lr = [0.01, 0.001]
hyper_param_labmda = [1, 2, 3]
hyper_param_gamma = [0, 0.1, 0.2, 0.3]
hyper_num_boost_round = [5, 10, 15]
threshold = 0.5

results = {}

for hyper_lr in tqdm(hyper_param_lr, desc= 'l_rate'):

    for hyper_depth in tqdm(hyper_param_depth, desc= 'depth'):
        
        for hyper_labmda in tqdm(hyper_param_labmda, desc= 'lambda'):
            
            for hyper_gamma in tqdm(hyper_param_gamma, desc= 'gamma'):

                skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20)

                val_scores = []
                val_mse_loss = []
                val_auroc_list = []
                val_auprc_list = []
                
                test_scores = []
                test_mse_loss = []
                test_auroc_list = []
                test_auprc_list = []

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

                        params = {'objective':'binary:logistic',
                                'eval_metric': 'logloss',
                                'tree_method': 'gpu_hist', 
                                'gpu_id': '2',
                                'learning_rate': hyper_lr, 
                                'max_depth': hyper_depth,
                                'lambda': hyper_labmda,
                                'gamma': hyper_gamma}

                        model_xgb = xgb.train(params, dtrain, 
                                            num_boost_round=20000, 
                                            evals=[(dvalidation, 'validation')], 
                                            verbose_eval=0, 
                                            early_stopping_rounds=1000)
                        
                        val_score = accuracy_score(y_validation, model_xgb.predict(dvalidation) > threshold)
                        val_loss = log_loss(y_true=y_validation, y_pred=model_xgb.predict(dvalidation))
                        val_auroc = roc_auc_score(y_true=y_validation, y_score=model_xgb.predict(dvalidation))
                        val_auprc = average_precision_score(y_true=y_validation, y_score=model_xgb.predict(dvalidation))

                        val_scores.append(val_score)
                        val_mse_loss.append(val_loss)
                        val_auroc_list.append(val_auroc)
                        val_auprc_list.append(val_auprc)
                        
                        test_score = accuracy_score(y_test, model_xgb.predict(dtest) > threshold)
                        test_loss = log_loss(y_true=y_test, y_pred=model_xgb.predict(dtest))
                        test_auroc = roc_auc_score(y_true=y_test, y_score=model_xgb.predict(dtest))
                        test_auprc = average_precision_score(y_true=y_test, y_score=model_xgb.predict(dtest))

                        test_scores.append(test_score)
                        test_mse_loss.append(test_loss)
                        test_auroc_list.append(test_auroc)
                        test_auprc_list.append(test_auprc)

                result = {'max_depth': hyper_depth, 
                          'learning_rate': hyper_lr, 
                          'lambda':hyper_labmda, 
                          'gamma': hyper_gamma, 
                          ## Valdiation Results
                          'val_scores':val_scores, 
                          'val_mean_score':np.mean(val_scores), 
                          'val_std_score':np.std(val_scores), 
                          'val_mse_loss':val_mse_loss, 
                          'val_mean_mse':np.mean(val_mse_loss),
                          'val_std_loss':np.std(val_mse_loss), 
                          'val_auroc':val_auroc_list,
                          'val_mean_auroc':np.mean(val_auroc_list), 
                          'val_stdf_auroc':np.std(val_auroc_list),
                          'val_auprc':val_auprc_list,
                          'val_mean_auprc':np.mean(val_auprc_list), 
                          'val_stdf_auprc':np.std(val_auprc_list),
                          ## Test Results
                          'test_scores':test_scores, 
                          'test_mean_score':np.mean(test_scores), 
                          'test_std_score':np.std(test_scores), 
                          'test_mse_loss':test_mse_loss, 
                          'test_mean_mse':np.mean(test_mse_loss),
                          'test_std_loss':np.std(test_mse_loss), 
                          'test_auroc':test_auroc_list,
                          'test_mean_auroc':np.mean(test_auroc_list), 
                          'test_stdf_auroc':np.std(test_auroc_list),
                          'test_auprc':test_auprc_list,
                          'test_mean_auprc':np.mean(test_auprc_list), 
                          'test_stdf_auprc':np.std(test_auprc_list),
                          }
                
                
                results["depth_" + str(hyper_depth) + "_lr_" + str(hyper_lr) + "_lambda_" + str(hyper_labmda) + "_gamma_" + str(hyper_gamma)] = result
# %%
with open('./result/results_mbpt_txt_baseline.pickle', 'wb') as file_nm:
    pickle.dump(results, file_nm, protocol=pickle.HIGHEST_PROTOCOL)
# %%
