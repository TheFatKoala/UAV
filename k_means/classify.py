import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# https://www.kaggle.com/smeschke/four-shapes
letter_df = pd.read_csv(r"C:\Users\hi2kh\OneDrive\Documents\GitHub\Machine-Learning\letter_classifier\letter_data.csv",
                 names=[str(i) for i in range(785)])

letter_df['0'] = 0.0  # assign label of letter_df to 0

