import numpy as np
import pandas as pd
import joblib

diabetes_data = pd.read_csv('Data\diabetes_binary.csv')
input = diabetes_data.sample(5)

history, reference_features, label = joblib.load('model_1\diabetes_model.pkl')
