# %%
import pandas as pd
import datatable
import numpy as np
import re
from IPython.display import display
pd.set_option("display.max_columns", None)
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

# %%
def baseline_parse(x):
    if x == "nan":
        return np.nan
    else:
        baseline_pattern = re.compile(r"\s*BASE\s*LINE\s*[|]\s*(?P<FVCC_90>\d*[.]*\d*)\s*(?P<FEV1C_90>\d*[.]*\d*)\s*(?P<FEF25_75C_90>\d*[.]*\d*)\s*")
        
        matched = re.search(baseline_pattern, x)
    
        return matched.group('FVCC_90'), matched.group('FEV1C_90'), matched.group('FEF25_75C_90')

def saline_parse(x):
    if x == "nan":
        return np.nan
    else:
        saline_pattern = re.compile(r"\s*SALINE\s*[|]\s*(?P<FVC_90>\d*[.]*\d*)\s*(?P<FEV1_90>\d*[.]*\d*)\s*(?P<FEF25_75_90>\d*[.]*\d*)\s*(?P<FVC_180>\d*[.]*\d*)\s*(?P<FEV1_180>\d*[.]*\d*)\s*(?P<FEF25_75_180>\d*[.]*\d*)\s*")
        
        matched = re.search(saline_pattern, x)
    
        return matched.group('FVC_90'), matched.group('FEV1_90'), matched.group('FEF25_75_90'), matched.group('FVC_180'), matched.group('FEV1_180'), matched.group('FEF25_75_180')
    
def methacholine_parse(x):
    if x == "nan":
        return np.nan
    else:
        saline_pattern = re.compile(r"\s*[(]*\s*(?P<portion>\d*[.]\d*)\s*[)]*\s*[|]\s*[(]*\s*(?P<FVC_90>\d*[.]*\d*)\s*[)]*\s*[(]*\s*(?P<FEV1_90>\d*[.]*\d*)\s*[)]*\s*[(]*\s*(?P<FEF25_75_90>\d*[.]*\d*)\s*[)]*\s*[(]*\s*(?P<FVC_180>\d*[.]*\d*)\s*[)]*\s*[(]*\s*(?P<FEV1_180>\d*[.]*\d*)\s*[)]*\s*[(]*\s*(?P<FEF25_75_180>\d*[.]*\d*)\s*[)]*\s*")
        
        matched = re.findall(saline_pattern, x)
        
        return matched
    
def assign_all_pattern(df):
    
    data = df.copy()
    
    extrated_data_baseline = data['Result_MBPT'].apply(lambda x: baseline_parse(str(x)))
    extrated_data_saline = data['Result_MBPT'].apply(lambda x: saline_parse(str(x)))
    extrated_data_methacholine = data['Result_MBPT'].apply(lambda x: methacholine_parse(str(x)))
    
    for patient, values in enumerate(extrated_data_baseline):
        
        if type(values) == float:
            continue
        else:
            columns = ['baseline_FVCC_90', 'baseline_FEV1C_90', 'baseline_FEF25_75C_90']
            for index, column in enumerate(columns, start=0):
                data.loc[patient, column] = values[index]
    
    for patient, values in enumerate(extrated_data_saline):
        
        if type(values) == float:
            continue
        else:
            columns = ['saline_FVC_90', 'saline_FEV1_90', 'saline_FEF25_75_90', 'saline_FVC_180', 'saline_FEV1_180', 'saline_FEF25_75_180']
            for index, column in enumerate(columns, start=0):
                data.loc[patient, column] = values[index]
    
    for patient, values in enumerate(extrated_data_methacholine):
        
        if type(values) == float:
            continue
        else:
            for value in values:
                columns = [str(value[0]) + '_FVC_90', str(value[0]) + '_FEV1_90', str(value[0]) + '_FEF25_75_90', str(value[0]) + '_FVC_180', str(value[0]) + '_FEV1_180', str(value[0]) + '_FEF25_75_180']
                for index, column in enumerate(columns, start=1):
                    data.loc[patient, column] = value[index]
    
    return data
    
# %%

if __name__ == '__main__':
    df_mbpt = pd.read_excel("../data/AI_asthmaDx 20210724.xlsx")
    df_result = assign_all_pattern(df_mbpt)
    df_result.to_csv('../data/AI_asthmaDX_parsed.csv', encoding='utf-8-sig', index=False)
# %%
