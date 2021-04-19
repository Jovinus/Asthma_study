# %%
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)

# %%
df_orig = pd.read_excel('../data/추가결과_이진영_20180907.xlsx', sheet_name='db')
display(df_orig.head())
# %%
df_orig.info()
# %%
df_orig.describe()

# %%
col = ['age', 'sex', 'FeNO', 'sp_eosinophil', 'mbpt', 'pc20', 'asthma']
df_orig[col]

# %%
import seaborn as sns
plt.figure(figsize=(15, 15))
sns.pairplot(df_orig[col])
plt.show()

# %%
from sklearn.model_selection import train_test_split
col_m = ['FeNO', 'sp_eosinophil', 'mbpt', 'pc20']
y = df_orig['asthma'].values
X = df_orig[col_m].values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1004, test_size=0.3)

# %%
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(criterion='gini', random_state=1004, max_depth = 3)
dt_clf.fit(X_train, y_train)
print(dt_clf.score(X_train, y_train))
print(dt_clf.score(X_test, y_test))

# %%
from sklearn.ensemble import RandomForestClassifier
rt_clf = RandomForestClassifier(criterion='gini', random_state=1004, 
                                n_estimators=100, n_jobs=-1)
rt_clf.fit(X_train, y_train)
print(rt_clf.score(X_train, y_train))
print(rt_clf.score(X_test, y_test))

# %%
from sklearn import tree
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dt_clf, 
                   feature_names=col_m,  
                   class_names='TF',
                   filled=True)
fig.show()

# %%
