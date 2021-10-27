# %%
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, auc, roc_curve, precision_recall_curve
pd.set_option("display.max_columns", None)

def performances_hard_decision(y_test, y_proba, threshold_of_interest=0.5, youden=False):
    
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    
    if(youden):
        threshold_of_interest = thresholds[np.argmax(tpr - fpr)]
    
    y_pred = y_proba >= threshold_of_interest
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    ppv = tp / (tp+fp)
    sensitivity = tp / (tp+fn)
    specificity = tn / (tn+fp)
    npv = tn / (tn+fn)
    accuracy = (tp+tn) / (tp+tn+fp+fn)
    f1 = (2 * ppv * sensitivity) / (ppv + sensitivity)
    
    return specificity, sensitivity, ppv, npv, f1, accuracy, threshold_of_interest, roc_auc, pr_auc

# %% Boot straping 
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression

def cutoff_analysis(X, y, data, youden=True):
    n_iteration = 1000
    test_size = 0.3
    n_size = int(len(data) * test_size)

    data = data[[X, y]].values

    # run bootstrap
    stats_specificity = list()
    stats_sensitivity = list()
    stats_ppv = list()
    stats_npv = list()
    stats_f1 = list()
    stats_accuracy = list()
    stats_threshold = list()
    stats_threshold_per = list()
    stats_auroc = list()
    stats_auprc = list()

    for i in range(n_iteration):
        train = resample(data, n_samples=n_size, stratify=data[:, -1])
        test = np.array([x for x in data if x.tolist() not in train.tolist()])
        
        model = LogisticRegression()
        model.fit(train[:, :-1], train[:, -1])
        
        y_hat_test = model.predict_proba(test[:, :-1])
        specificity, sensitivity, ppv, npv, f1, accuracy, threshold_of_interest, roc_auc, pr_auc = performances_hard_decision(test[:, -1], y_hat_test[:, 1], youden=youden)
        
        threshold = (np.log(threshold_of_interest/(1-threshold_of_interest)) - model.intercept_) /model.coef_
        
        stats_specificity.append(specificity)
        stats_sensitivity.append(sensitivity)
        stats_ppv.append(ppv)
        stats_npv.append(npv)
        stats_f1.append(f1)
        stats_accuracy.append(accuracy)
        stats_threshold.append(threshold)
        stats_threshold_per.append(threshold_of_interest)
        stats_auprc.append(roc_auc)
        stats_auroc.append(pr_auc)

    metrics = [{'specificity':stats_specificity}, {'sensitivity':stats_sensitivity}, 
            {'ppv':stats_ppv}, {'npv':stats_npv}, {'f1':stats_f1}, {'accuracy':stats_accuracy}, 
            {'threshold':stats_threshold}, {'threshold_per':stats_threshold_per}, {'auprc':stats_auprc}, {'auroc':stats_auroc}]
    for metric in metrics:
        name, metric = list(metric.items())[0]
        mean = np.mean(metric)
        alpha = 0.95
        p = ((1.0-alpha)/2.0) * 100
        lower = max(0.0, np.percentile(metric, p))
        p = (alpha+((1.0-alpha)/2.0)) * 100
        if name == 'threshold':
            upper = max(1.0, np.percentile(metric, p))
        else: 
            upper = min(1.0, np.percentile(metric, p))
        print(f'Mean {name} = {mean:.2f}, {alpha*100:.0f}% confidence interval {lower:3.2f} and {upper:3.2f}')
# %%
if __name__ == '__main__':
    
    df_orig = pd.read_excel('../data/20210913 ISE-FENO eos.xlsx', 
                            sheet_name='669 (77 missing eos제외)')
    print('Blood Eosinophil Counts')
    cutoff_analysis(X='Lab_EosCount', y='ISE_Eo3%', data=df_orig, youden=True)

    print('\nFeNO')
    cutoff_analysis(X='FeNO', y='ISE_Eo3%', data=df_orig, youden=True)
# %%
