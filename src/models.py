#models
#baseline 2

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

CAT_COLS = ['country', 'store', 'product']
NUM_COLS = ['month', 'dayofweek', 'dayofyear', 'is_weekend']


def train_ridge(train_df, val_df, alpha=1.0):
    #Trains the Ridge model using a log1p-transformed target and converts validation predictions back to the original scale.
    X_tr = pd.get_dummies(train_df[CAT_COLS + NUM_COLS], columns=CAT_COLS)
    X_va = pd.get_dummies(val_df[CAT_COLS + NUM_COLS], columns=CAT_COLS)
    X_va = X_va.reindex(columns=X_tr.columns, fill_value=0)

    y_tr = np.log1p(train_df['num_sold'])

    model = Ridge(alpha=alpha)
    model.fit(X_tr, y_tr)
    return np.expm1(model.predict(X_va))