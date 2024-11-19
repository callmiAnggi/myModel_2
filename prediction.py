import numpy as np
import pandas as pd
import joblib

diabetes_data = pd.read_csv('data\diabetes_binary.csv')
diabetes_data = diabetes_data.astype('int64')


pipeline, reference_features, label = joblib.load('testing1.pkl')

#input_data = diabetes_data[diabetes_data[label] == 1]
input_data = diabetes_data
input = input_data.sample(10)

X_new = input[reference_features]
Y_new = input[label]
prediction = pipeline.predict(X_new)

probabilities = pipeline.predict_proba(X_new)


categories = {
    "No Diabetes": lambda prob: prob < 0.45,
    "Pre-Diabetes": lambda prob: 0.45 <= prob <= 0.65,
    "Diabetes": lambda prob: prob > 0.65
}

for i, prob in enumerate(probabilities):
    positive_class_prob = prob[1]  
    for category, condition in categories.items():
        if condition(positive_class_prob):
            print(f"Sample {i + 1}: {category}")
            break

X_new.shape
