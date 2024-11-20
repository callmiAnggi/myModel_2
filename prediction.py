import numpy as np
import pandas as pd
import joblib

#diabetes_data = pd.read_csv('data\diabetes_binary.csv')
diabetes_data = pd.read_pickle('data\k_test.pkl')
diabetes_data = diabetes_data.astype('int64')


#pipeline, reference_features, label = joblib.load('testing1.pkl')
model, reference_features, label = joblib.load('no_scaling2.pkl')

#input_data = diabetes_data[diabetes_data[label] == 0]
input_data = diabetes_data
input = input_data.sample(10)

X_new = input.iloc[:, :-1]
y_new = input.iloc[:, -1]
#Y_new = input[reference_features]
#Y_new = input[label]
prediction = model.predict(X_new)
predicted_classes = (prediction > 0.5).astype(int)
#probabilities = model.predict_proba(X_new)


categories = {
    "No Diabetes": lambda prob: prob < 0.4,
    "Pre-Diabetes": lambda prob: 0.4 <= prob <= 0.6,
    "Diabetes": lambda prob: prob > 0.6
}

for i, prob in enumerate(prediction):
    positive_class_prob = prob[1]  
    for category, condition in categories.items():
        if condition(positive_class_prob):
            print(f"Sample {i + 1}: {category}")
            break

X_new.shape
