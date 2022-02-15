

#------ lib packages 

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

#----------- external imports 


def split_data(data_type:str):

    if data_type == 'binary': 
        df_heart = pd.read_csv("data/heart.csv")

        le=LabelEncoder()

        df_heart['Sex']=le.fit_transform(df_heart['Sex'])
        df_heart['RestingECG']=le.fit_transform(df_heart['RestingECG'])
        df_heart['ChestPainType']=le.fit_transform(df_heart['ChestPainType'])
        df_heart['ExerciseAngina']=le.fit_transform(df_heart['ExerciseAngina'])
        df_heart['ST_Slope']=le.fit_transform(df_heart['ST_Slope'])

        X = df_heart.drop('HeartDisease', axis=1)
        y = df_heart['HeartDisease']

        y = np.where(y==0,-1,1)

    if data_type == 'multiclass': 
        df_heart = pd.read_csv("data/heart.csv")

        le=LabelEncoder()

        df_heart['Sex']=le.fit_transform(df_heart['Sex'])
        df_heart['RestingECG']=le.fit_transform(df_heart['RestingECG'])
        df_heart['ChestPainType']=le.fit_transform(df_heart['ChestPainType'])
        df_heart['ExerciseAngina']=le.fit_transform(df_heart['ExerciseAngina'])
        df_heart['ST_Slope']=le.fit_transform(df_heart['ST_Slope'])

        X = df_heart.drop('HeartDisease', axis=1)
        y = df_heart['HeartDisease']

        y = np.where(y==0,-1,1)

    return X, y