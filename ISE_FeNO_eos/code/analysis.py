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
   
    print("AUROC: {:.3f}".format(roc_auc))
   
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    
    print("AUPRC: {:.3f}".format(pr_auc))

    
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
   
    print("specificity: {:.3f}".format(specificity))
    print("sensitivity: {:.3f}".format(sensitivity))
    print("PPV: {:.3f}".format(ppv))
    print("NPV: {:.3f}".format(npv))
    print("f1: {:.3f}".format(f1))
    print("accuracy: {:.3f}".format(accuracy))
    print("threshold: {:.3f}".format(threshold_of_interest))

# %%
df_orig = pd.read_excel('../data/20210913 ISE-FENO eos.xlsx', 
                        sheet_name='669 (77 missing eos제외)')
display(df_orig.head())
# %%

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.histplot(data=df_orig ,x = 'Lab_EosCount', ax=ax, kde=True, hue='ISE_Eo3%')
plt.xlim(0, 1500)
plt.show()
# %%

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.histplot(data=df_orig ,x = 'FeNO', ax=ax, kde=True, hue='ISE_Eo3%')
# plt.xlim(0, 1000)
plt.show()
# %%

from sklearn.linear_model import LogisticRegression

logit_bl = LogisticRegression()
logit_bl.fit(df_orig[['Lab_EosCount']], df_orig[['ISE_Eo3%']])

# %%
y_proba = logit_bl.predict_proba(df_orig[['Lab_EosCount']])[:, 1]
performances_hard_decision(df_orig['ISE_Eo3%'], y_proba, youden=True)
# %%
from sklearn.linear_model import LogisticRegression

X = df_orig[['Lab_EosCount']].values
y = df_orig['ISE_Eo3%'].values

logistic = LogisticRegression()

logistic.fit(X, y)

print("Accuracy = {}".format(logistic.score(X, y)))

y_proba = logistic.predict_proba(X)[:, 1]

from sklearn import metrics
# calculate fpr, tpr, and thresholds
fpr, tpr, thresholds = metrics.roc_curve(y_true=y, y_score=y_proba)

sensitivity = tpr
specificity = 1 - fpr
summation = sensitivity + specificity

bl_eosinophil = ((np.log(thresholds/(1-thresholds)) - logistic.intercept_) /logistic.coef_[:, 0]).reshape(-1)

accuracy = np.zeros_like(bl_eosinophil)

for index, value in enumerate(bl_eosinophil):
    if index == 0:
        continue
    else:
        accuracy[index] = metrics.accuracy_score((df_orig['Lab_EosCount'] >= value).astype(int), df_orig['ISE_Eo3%'].values)



results = pd.DataFrame({'specificity':specificity, 
              'sensitivity':sensitivity, 
              'summation':summation,
              'Lab_EosCount':bl_eosinophil,
              'thresholds':thresholds, 
              'accuracy': accuracy})

results = results.assign(sum_sen_spec = lambda x: x['sensitivity'] + x['specificity'])

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.lineplot(data=results, x='Lab_EosCount', y='sensitivity', ax=ax, label='Sensitivity')
sns.lineplot(data=results, x='Lab_EosCount', y='specificity', ax=ax, label='Specificity')
sns.lineplot(data=results, x='Lab_EosCount', y='summation', ax=ax, label='Sum_Sen_Spec')
plt.ylabel("Score")
plt.axvline(x=results.iloc[np.where(results['summation'] == results['summation'].max())]['Lab_EosCount'].values, color='r')
plt.xlim(0, 1000)
plt.legend()
plt.show()

print(results.iloc[np.where(results['summation'] == results['summation'].max())]['Lab_EosCount'])
# %%
from sklearn.linear_model import LogisticRegression

X = df_orig[['FeNO']].values
y = df_orig['ISE_Eo3%'].values

logistic = LogisticRegression()

logistic.fit(X, y)

print("Accuracy = {}".format(logistic.score(X, y)))

y_proba = logistic.predict_proba(X)[:, 1]
performances_hard_decision(y, y_proba, youden=True)

from sklearn import metrics
# calculate fpr, tpr, and thresholds
fpr, tpr, thresholds = metrics.roc_curve(y_true=y, y_score=y_proba)

sensitivity = tpr
specificity = 1 - fpr
summation = sensitivity + specificity

bl_eosinophil = ((np.log(thresholds/(1-thresholds)) - logistic.intercept_) /logistic.coef_[:, 0]).reshape(-1)

accuracy = np.zeros_like(bl_eosinophil)

for index, value in enumerate(bl_eosinophil):
    if index == 0:
        continue
    else:
        accuracy[index] = metrics.accuracy_score((df_orig['FeNO'] >= value).astype(int), df_orig['ISE_Eo3%'].values)



results = pd.DataFrame({'specificity':specificity, 
              'sensitivity':sensitivity, 
              'summation':summation,
              'FeNO':bl_eosinophil,
              'thresholds':thresholds, 
              'accuracy': accuracy})

results = results.assign(sum_sen_spec = lambda x: x['sensitivity'] + x['specificity'])

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.lineplot(data=results, x='FeNO', y='sensitivity', ax=ax, label='Sensitivity')
sns.lineplot(data=results, x='FeNO', y='specificity', ax=ax, label='Specificity')
plt.ylabel("Score")
sns.lineplot(data=results, x='FeNO', y='summation', ax=ax, label='Sum_Sen_Spec')
plt.axvline(x=results.iloc[np.where(results['summation'] == results['summation'].max())]['FeNO'].values, color='r')
# plt.xlim(0, 1000)
plt.legend()
plt.show()

print(results.iloc[np.where(results['summation'] == results['summation'].max())]['FeNO'])
# %%
## Rank Correlation Coefficients Analysis

df_orig['ISE_Eo3%']
df_orig = df_orig.assign(bl_cut = lambda x: np.where(x['Lab_EosCount'] >= 184, 1, 0), 
                         feno_cut = lambda x: np.where(x['FeNO'] >= 45, 1, 0))

from scipy.stats import spearmanr
tau, p_value = spearmanr(df_orig['Lab_EosCount'], df_orig['ISE_Eo3%'])
print(tau, p_value)

tau, p_value = spearmanr(df_orig['FeNO'], df_orig['ISE_Eo3%'])
print(tau, p_value)


# %% Boot straping 
