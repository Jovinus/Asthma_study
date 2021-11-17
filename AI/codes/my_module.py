import numpy as np
from sklearn.metrics import confusion_matrix, auc, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt


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