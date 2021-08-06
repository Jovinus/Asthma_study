# %%
import pandas as pd
import datatable
import numpy as np
import re
from IPython.display import display
# %%
df_mbpt = pd.read_excel("../data/AI_asthmaDx 20210724.xlsx")
# %%

## SPLIT PATTERN ##

## Baseline
re.compile(r"\s*BASE\s*LINE\s*[|]\s*(?P<FVCC_90>\d*[.]*\d*)\s*(?P<FEV1C_90>\d*[.]*\d*)\s*(?P<FEF25_75C_90>\d*[.]*\d*)\s*")

## Saline
re.compile(r"\s*SALINE\s*[|]\s*(?P<FVC_90>\d*[.]*\d*)\s*(?P<FEV1_90>\d*[.]*\d*)\s*(?P<FEF25_75_90>\d*[.]*\d*)\s*(?P<FVC_180>\d*[.]*\d*)\s*(?P<FEV1_180>\d*[.]*\d*)\s*(?P<FEF25_75_180>\d*[.]*\d*)\s*")

## METHACHOLINE
re.compile(r"\s*METHACHOLINE\s*[(]*\s*(?P<portion>\d*[.]\d*)\s*[)]*\s*[|]\s*[(]*\s*(?P<FVC_90>\d*[.]*\d*)\s*[)]*\s*[(]*\s*(?P<FEV1_90>\d*[.]*\d*)\s*[)]*\s*[(]*\s*(?P<FEF25_75_90>\d*[.]*\d*)\s*[)]*\s*[(]*\s*(?P<FVC_180>\d*[.]*\d*)\s*[)]*\s*[(]*\s*(?P<FEV1_180>\d*[.]*\d*)\s*[)]*\s*[(]*\s*(?P<FEF25_75_180>\d*[.]*\d*)\s*[)]*\s*")


# %%
r"\s*[(]*\s*(\d*[.]\d*)\s*[)]*\s*[|]\s*[(]*\s*(\d*[.]*\d*)\s*[)]*\s*[(]*\s*(\d*[.]*\d*)\s*[)]*\s*[(]*\s*(\d*[.]*\d*)\s*[)]*\s*[(]*\s*(\d*[.]*\d*)\s*[)]*\s*[(]*\s*(\d*[.]*\d*)\s*[)]*\s*[(]*\s*(\d*[.]*\d*)\s*[)]*\s*"