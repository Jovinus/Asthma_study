# %% Import Library to Use
import numpy as np
import re
import pandas as pd
pd.set_option("display.max_columns", None)
# %%
df_data = pd.read_excel("../data/asthma_dataset_final.xlsx", sheet_name="746 (소아2 중복3 missing 8제거)")

# %%
baseline = ['MBPT_result', 'maxFall_FEV1_p', 'PC20_32']

baseline_txt = [i for i in df_data.columns if "baseline" in i]
salline_txt = [i for i in df_data.columns if "saline" in i]

subs = r"0.05|0.5|2.0|8.0|16.0|32.0"
mbpt_txt = [x for x in df_data.columns if re.search(subs, str(x))]

# %%
