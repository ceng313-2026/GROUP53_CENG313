
import pandas as pd

def load_data(data_dir):
    #Reads train, test and sample submission files. 
    train = pd.read_csv(data_dir + 'train.csv', parse_dates=['date'])
    test = pd.read_csv(data_dir + 'test.csv', parse_dates=['date'])
    sample_sub = pd.read_csv(data_dir + 'sample_submission.csv')
    return train, test, sample_sub

def add_date_features(df):
    #Extracts features from the date column.

    df = df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    return df

