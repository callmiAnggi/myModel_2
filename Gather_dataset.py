import pandas as pd

diabetes_data = pd.read_csv('Data\diabetes_binary.csv')
diabetes_data.info()

selected_features = [
    'HighBP',
    'BMI',
    'GenHlth',
    'DiffWalk', 
    'HighChol', 
    'Age', 
    'HeartDiseaseorAttack', 
    'PhysHlth', 
    'Stroke', 
    'MentHlth',
    'Diabetes_binary'
]

diabetes_data = diabetes_data[selected_features]
diabetes_data

diabetes_data = diabetes_data.astype('int64')

diabetes_data.to_pickle('Data/diabetes_binary.pkl')

