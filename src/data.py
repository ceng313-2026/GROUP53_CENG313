
import pandas as pd

def load_data(data_dir):
    #Reads train, test and sample submission files. 
    train = pd.read_csv(data_dir + 'train.csv', parse_dates=['date'])
    test = pd.read_csv(data_dir + 'test.csv', parse_dates=['date'])
    sample_sub = pd.read_csv(data_dir + 'sample_submission.csv')
    return train, test, sample_sub