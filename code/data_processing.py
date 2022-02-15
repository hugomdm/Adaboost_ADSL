#------ lib packages 

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris

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

        plt.figure(figsize=(8,6))
        ax = sns.countplot(x=df_heart['HeartDisease'],
                        order=df_heart['HeartDisease'].value_counts(ascending=True).index);
                
        abs_values = df_heart['HeartDisease'].value_counts(ascending=True)
        rel_values = df_heart['HeartDisease'].value_counts(ascending=True, normalize=True).values * 100
        lbls = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values, rel_values)]

        ax.bar_label(container=ax.containers[0], labels=lbls)
        ax.set_xlabel('Heart Disease')
        plt.savefig('../img/dist_heart.png')

        features = data.loc[:, :'ST_Slope']
        tsne = TSNE(n_components=2, random_state=0)
        projections = tsne.fit_transform(features)

        fig = px.scatter(
            projections, x=0, y=1,
            color=data.HeartDisease, labels={'color': 'HeartDisease'}
        )
        fig.savefig('../img/tnse_heart.png')

        X = df_heart.drop('HeartDisease', axis=1)
        y = df_heart['HeartDisease']

        y = np.where(y==0,-1,1)

    if data_type == 'multiclass': 
        data = load_iris()

        X = data.data
        y = data.target

    return X, y