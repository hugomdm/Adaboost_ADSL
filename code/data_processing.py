#------ lib packages 

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.express as px
from imblearn.under_sampling import RandomUnderSampler

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
        plt.savefig('../img/dist_heart.png', bbox_inches='tight')

        features = df_heart.loc[:, :'ST_Slope']
        tsne = TSNE(n_components=2, random_state=0)
        projections = tsne.fit_transform(features)

        fig = px.scatter(
            projections, x=0, y=1,
            color=df_heart.HeartDisease, labels={'color': 'Target'},
            color_continuous_scale=px.colors.sequential.Viridis
        )
        fig.update_layout(
            margin=dict(l=5, r=5, t=20, b=20),
        )

        fig.write_image("../img/tnse_heart.png")


        X = df_heart.drop('HeartDisease', axis=1)
        y = df_heart['HeartDisease']

        y = np.where(y==0,-1,1)

    if data_type == 'multiclass': 
        data_train = pd.read_excel("data/ann-train.xlsx", header=None)
        data_test = pd.read_excel("data/ann-test.xlsx", header=None)

        columns = ['age', 'sex','on_thyroxine','query_on_thyroxine','on_antithyroid_medication','sick', \
           'pregnant','thyroid_surgery','I131_treatment', \
           'query_hypothyroid','query_hyperthyroid', 'lithium', 'goitre', \
            'tumor','hypopituitary', 'psych', 'TSH', 'T3', 'TTT4', 'T4U', 'FTI', 'Target']
        data_train.columns = columns
        data_test.columns = columns
        data = pd.concat([data_train, data_test], ignore_index=True, sort=False)

        le=LabelEncoder()

        data['Target']=le.fit_transform(data['Target'])

        X = data.drop('Target', axis=1)
        y = np.array(data['Target'])

        under = RandomUnderSampler()
        X, y = under.fit_resample(X, y)

        tsne = TSNE(n_components=2, random_state=0)
        projections = tsne.fit_transform(X)

        fig = px.scatter(
            projections, x=0, y=1,
            color=y, labels={'color':'Target'}, 
            color_continuous_scale=px.colors.sequential.Viridis
        )
        fig.update_layout(
        margin=dict(l=5, r=5, t=20, b=20),
        )
        fig.write_image("../img/tnse_tyroid.png")

    return X, y