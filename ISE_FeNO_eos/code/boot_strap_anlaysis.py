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
    
    return specificity, sensitivity, ppv, npv, f1, accuracy, threshold_of_interest, roc_auc, pr_auc, fpr, tpr, precision, recall

def plot_roc_curve(x, y, data, mean, std):
    x = np.array(data[x]).reshape(-1, 100)
    y = np.array(data[y]).reshape(-1, 100)
    
    mean_tpr = y.mean(axis=0).reshape(-1)
    mean_tpr[-1] = 1.0
    mean_fpr = x.mean(axis=0).reshape(-1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.grid()
    # sns.lineplot(x=x, y=y, data=data, ax=ax, label=f'AUROC = {mean:.2f} $\pm$ {std:.2f}')
    ax.plot(mean_fpr, mean_tpr, label=f'AUROC = {mean:.2f} $\pm$ {std:.2f}')
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    
    std_tpr = y.std(axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    plt.legend()
    plt.show()

def plot_prc_curve(x, y, data, mean, std):
    x = np.array(data[x]).reshape(-1, 100)
    y = np.array(data[y]).reshape(-1, 100)
    
    mean_tpr = y.mean(axis=0).reshape(-1)
    mean_tpr[-1] = 0
    mean_fpr = x.mean(axis=0).reshape(-1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.grid()
    # sns.lineplot(x=x, y=y, data=data, ax=ax, label=f'AUROC = {mean:.2f} $\pm$ {std:.2f}')
    ax.plot(mean_fpr, mean_tpr, label=f'AUPRC = {mean:.2f} $\pm$ {std:.2f}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    
    std_tpr = y.std(axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    plt.legend()
    plt.show()

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
    
    fig_fpr = list()
    fig_tpr = list()
    fig_precision = list()
    fig_recall = list()
    
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)
    
    for i in range(n_iteration):
        train = resample(data, n_samples=n_size, stratify=data[:, -1])
        test = np.array([x for x in data if x.tolist() not in train.tolist()])
        
        model = LogisticRegression()
        model.fit(train[:, :-1], train[:, -1])
        
        y_hat_test = model.predict_proba(test[:, :-1])
        specificity, sensitivity, ppv, npv, f1, accuracy, threshold_of_interest, roc_auc, pr_auc, fpr, tpr, precision, recall = performances_hard_decision(test[:, -1], y_hat_test[:, 1], youden=youden)
        
        threshold = (np.log(threshold_of_interest/(1-threshold_of_interest)) - model.intercept_) /model.coef_
        
        stats_specificity.append(specificity)
        stats_sensitivity.append(sensitivity)
        stats_ppv.append(ppv)
        stats_npv.append(npv)
        stats_f1.append(f1)
        stats_accuracy.append(accuracy)
        stats_threshold.append(threshold)
        stats_threshold_per.append(threshold_of_interest)
        stats_auprc.append(pr_auc)
        stats_auroc.append(roc_auc)
        
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        
        interp_precision = np.interp(mean_recall, np.flip(recall), np.flip(precision))
        # interp_precision[0] = 1.0
        
        fig_fpr.extend(mean_fpr.tolist())
        fig_tpr.extend(interp_tpr.tolist())
        fig_precision.extend(interp_precision.tolist())
        fig_recall.extend(mean_recall.tolist())
        
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
    
    fig_auroc = pd.DataFrame({'fpr':fig_fpr, 'tpr':fig_tpr})
    fig_auprc = pd.DataFrame({'precision':fig_precision, 'recall':fig_recall})
    
    plot_roc_curve(x='fpr', y='tpr' , data=fig_auroc, mean=np.mean(stats_auroc), std=np.std(stats_auroc))
    plot_prc_curve(x='recall', y='precision' , data=fig_auprc, mean=np.mean(stats_auprc), std=np.std(stats_auprc))

def test_plot(x, y):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.grid()
    # sns.lineplot(x=x, y=y, data=data, ax=ax, label=f'AUROC = {mean:.2f} $\pm$ {std:.2f}')
    ax.plot(x, y)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()
# %%
if __name__ == '__main__':
    
    df_orig = pd.read_excel('../data/20210913 ISE-FENO eos.xlsx', 
                            sheet_name='669 (77 missing eos제외)')
    print('Blood Eosinophil Counts')
    cutoff_analysis(X='Lab_EosCount', y='ISE_Eo3%', data=df_orig, youden=True)

    print('\nFeNO')
    cutoff_analysis(X='FeNO', y='ISE_Eo3%', data=df_orig, youden=True)
# %%
    print('\nBlood Eosinophil Counts')
    cutoff_analysis(X='Lab_EosCount', y='ISE_Eo3%', data=df_orig.query('Asthma == 1'), youden=True)

    print('\nFeNO')
    cutoff_analysis(X='FeNO', y='ISE_Eo3%', data=df_orig.query('Asthma == 1'), youden=True)
    
    print('\nBlood Eosinophil Counts')
    cutoff_analysis(X='Lab_EosCount', y='ISE_Eo3%', data=df_orig.query('Asthma == 0'), youden=True)

    print('\nFeNO')
    cutoff_analysis(X='FeNO', y='ISE_Eo3%', data=df_orig.query('Asthma == 0'), youden=True)