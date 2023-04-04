import os
import pandas as pd

def load_data(path):
    df_nongeorges = pd.read_csv(os.path.join(path, 'non_georges.csv'),
                            header=None)
    
    df_georges = pd.read_csv(os.path.join(path, 'georges.csv'),
                            header=None)
    
    return df_nongeorges, df_georges