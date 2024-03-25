import pandas as pd


def add_to_csv(path, new_row):
    df = pd.read_csv(path)
    
    df.loc[len(df.index)] = new_row
    
    df.to_csv(path, index=False)