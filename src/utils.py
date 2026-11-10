import numpy as np


def mape(y_true, y_pred):
    #Mean Absolute Percentage Error
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def time_split(df, val_year=2016):
    #Splits the training and validation sets based on date and removes rows with missing num_sold values.
    df = df.dropna(subset=['num_sold'])
    train_part = df[df['date'].dt.year < val_year]
    val_part = df[df['date'].dt.year == val_year]
    return train_part, val_part