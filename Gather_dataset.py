import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

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
train_data, test_data = train_test_split(diabetes_data, test_size=0.2, random_state=42)

train_data.to_pickle('data/train.pkl')
test_data.to_pickle('data/test.pkl')

#diabetes_data.to_pickle('Data/diabetes_binary.pkl')

