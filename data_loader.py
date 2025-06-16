# data_loader.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data():
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    submission_df = pd.read_csv("data/sample_submission.csv")
    return train_df, test_df, submission_df