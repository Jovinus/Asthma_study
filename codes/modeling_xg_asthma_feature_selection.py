# %% Import package to use
import numpy as np
from numpy.lib.function_base import disp
import datatable
import pandas as pd
import xgboost as xgb
from IPython.display import display
pd.set_option("display.max_columns", None)
import os

# %% Load dataset
DATA_PATH = "/home/lkh256/Studio/Asthma"
df_init = datatable.fread(os.path.join(DATA_PATH, 'dataset.csv'), 
                          encoding='utf-8-sig', 
                          na_strings=['', 'NA']).to_pandas()

print("Number of samples = {}".format(len(df_init)))
display(df_init.head())

# %% Check missings
print("Check their is any missing variables in dataset: \n", df_init.isnull().sum())

# %%
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df_init,  
                                       random_state=1004, 
                                       stratify=df_init['asthma'], 
                                       test_size=0.2)

print("Train set size = {}".format(len(train_set)))
print("Test set size = {}".format(len(test_set)))
# %% Cross-Val Model

from tqdm import tqdm
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
import numpy as np
import pickle

hyper_param_depth = [3, 4, 5]
hyper_param_lr = [0.01, 0.001]
hyper_param_labmda = [1, 2, 3]
hyper_param_gamma = [0, 0.1, 0.2, 0.3]
hyper_num_boost_round = [5, 10, 15]
threshold = 0.5

results = {}

feature_onset = ['mbpt', 'FeNO']
feature_selection_set = ['sp_eosinophil', 'Bl_eos_count', 'maxFall_FEV1_percent', 'FEF2575_afterMBPT']

for feature_to_add in tqdm(feature_selection_set, desc='feature_comb'):
    feature_mask = feature_onset + [feature_to_add]

    for hyper_lr in tqdm(hyper_param_lr, desc='l_rate'):

        for hyper_depth in tqdm(hyper_param_depth, desc= 'depth'):
            
            for hyper_labmda in tqdm(hyper_param_labmda, desc= 'lambda'):
                
                for hyper_gamma in tqdm(hyper_param_gamma, desc= 'gamma'):

                    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1)

                    scores = []
                    mse_loss = []

                    for train_index, validation_index in skf.split(train_set, train_set['asthma']):
                            X_train = train_set.iloc[train_index][feature_mask].values
                            y_train = train_set.iloc[train_index]['asthma'].values
                            dtrain = xgb.DMatrix(X_train, label=y_train)
                            
                            X_validation = train_set.iloc[validation_index][feature_mask].values
                            y_validation = train_set.iloc[validation_index]['asthma'].values
                            dvalidation = xgb.DMatrix(X_validation, label=y_validation)
                            
                            X_test = test_set[feature_mask].values
                            y_test = test_set['asthma'].values
                            dtest = xgb.DMatrix(X_test, label=y_test)

                            params = {'objective':'binary:logistic',
                                'eval_metric': 'logloss',
                                'tree_method': 'gpu_hist', 
                                'gpu_id': '0',
                                'learning_rate': hyper_lr, 
                                'max_depth': hyper_depth,
                                'lambda': hyper_labmda,
                                'gamma': hyper_gamma}

                            model_xgb = xgb.train(params, dtrain, 
                                                num_boost_round=20000, 
                                                evals=[(dvalidation, 'validation')], 
                                                verbose_eval=0, 
                                                early_stopping_rounds=1000)
                            
                            score = accuracy_score(y_test, model_xgb.predict(dtest) > threshold)
                            loss = log_loss(y_true=y_test, y_pred=model_xgb.predict(dtest))

                            scores.append(score)
                            mse_loss.append(loss)

                    result = {'max_depth': hyper_depth, 
                            'learning_rate': hyper_lr, 
                            'lambda':hyper_labmda, 
                            'gamma': hyper_gamma, 
                            'scores':scores, 
                            'mean_score':np.mean(scores), 
                            'std_score':np.std(scores), 
                            'mse_loss':mse_loss, 
                            'mean_mse':np.mean(mse_loss),
                            'std_loss':np.std(mse_loss), 
                            'feature':'_'.join(feature_mask)}
                    # print(result['mean_score'])
                    results["depth_" + str(hyper_depth) + "_lr_" + str(hyper_lr) + "_lambda_" + str(hyper_labmda) + "_gamma_" + str(hyper_gamma) + "_feature_" + '_'.join(feature_mask)] = result
# %%
with open('../asthma_xg_classifier_subset_feature_selection.pickle', 'wb') as file_nm:
    pickle.dump(results, file_nm, protocol=pickle.HIGHEST_PROTOCOL)
# %%
