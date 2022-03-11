import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy import ndimage
import random

df = pd.read_csv(r"C:\Users\hi2kh\Documents\GitHub\Machine-Learning\letter_classifier\letter_data.csv",
                 names=[str(i) for i in range(785)])
