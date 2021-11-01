# %% 
import pandas as pd
from scipy.stats import ttest_ind

# %%
metrics_bl_asthma = pd.read_csv("../result/bl_cut_sub_a.csv")
metrics_feno_asthma = pd.read_csv("../result/feno_cut_sub_a.csv")
metrics_bl_healthy = pd.read_csv("../result/bl_cut_sub_h.csv")
metrics_feno_healthy = pd.read_csv("../result/feno_cut_sub_h.csv")
# %%

print(ttest_ind(metrics_bl_asthma['threshold'], metrics_bl_healthy['threshold']))
# %%
print(ttest_ind(metrics_feno_asthma['threshold'], metrics_feno_healthy['threshold']))
# %%
