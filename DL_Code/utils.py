import torch
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
pd.set_option("display.max_columns", None)

def load_dataset():
    DATA_PATH = "/home/lkh256/Studio/Asthma/data"
    df_orig = pd.read_excel(os.path.join(DATA_PATH, '추가결과_이진영_20180907.xlsx'), sheet_name='db')
    
    #### basic data preprocessing
    categorical = ['sex', 'mbpt']
    numeric = ['age', 'pc20', 'sp_eosinophil', 'FeNO']
    
    df_orig['sex'] = np.where(df_orig['sex'] == 'M', 0, 1)
    
    df_orig[categorical] = df_orig[categorical] * 1
    
    #### Define columns to analysis
    column_mask = numeric + categorical
    
    x = torch.from_numpy(df_orig[column_mask].values).float()
    y = torch.from_numpy(df_orig['asthma'].values).float()
    
    return x, y

def split_data(x, y, train_ratio=0.8, device=torch.device('cpu')):
    train_cnt = int(x.size(0) * train_ratio)
    valid_cnt = x.size(0) - train_cnt
    
    # Shuffle dataset to split into train/valid set
    indices = torch.randperm(x.size(0))
    
    x = torch.index_select(
        x, 
        dim=0, 
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)
    
    y = torch.index_select(
        y, 
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)
    
    scaler = StandardScaler()
    scaler.fit(x[0].numpy())
    x = (torch.from_numpy(scaler.transform(x[0].numpy())).to(device), 
         torch.from_numpy(scaler.transform(x[1].numpy())).to(device))
    y = (torch.reshape(y[0], (-1, 1)).to(device), torch.reshape(y[1], (-1, 1)).to(device))
    
    return x, y


def get_hidden_sizes(input_size, output_size, n_layers, n_node_first_hidden):
    # step_size = int((input_size - output_size) / n_layers)
    
    step_size = int((n_node_first_hidden - output_size) / n_layers)
    
    hidden_sizes =[]
    current_size = n_node_first_hidden
    
    for i in range(n_layers - 1):
        if i == 0:
            hidden_sizes += [current_size]
        else:
            hidden_sizes += [current_size - step_size]
        current_size = hidden_sizes[-1]
        
    return hidden_sizes

if __name__ == '__main__':
    x, y = load_dataset()
    x, y = split_data(x, y, train_ratio=0.8)
    print(x)
    print(y)
    
    print(get_hidden_sizes(100, 1, 3, 100))