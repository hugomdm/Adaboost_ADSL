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

#----------- code


def process_data_binary():
    """
        Function to load, pre-process and separate into X and y the Hearts data
        Return: 
            X: features that will be used in training 
            y: target feature 
    """
    #loading data
    df_heart = pd.read_csv("data/heart.csv")

    #turning categorical values into numerical
    le=LabelEncoder()
    df_heart['Sex']=le.fit_transform(df_heart['Sex'])
    df_heart['RestingECG']=le.fit_transform(df_heart['RestingECG'])
    df_heart['ChestPainType']=le.fit_transform(df_heart['ChestPainType'])
    df_heart['ExerciseAngina']=le.fit_transform(df_heart['ExerciseAngina'])
    df_heart['ST_Slope']=le.fit_transform(df_heart['ST_Slope'])

    #ploting the distribution of the classes and saving
    plt.figure(figsize=(8,6))
    ax = sns.countplot(x=df_heart['HeartDisease'],
                    order=df_heart['HeartDisease'].value_counts(ascending=True).index);
    abs_values = df_heart['HeartDisease'].value_counts(ascending=True)
    rel_values = df_heart['HeartDisease'].value_counts(ascending=True, normalize=True).values * 100
    lbls = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values, rel_values)]
    ax.bar_label(container=ax.containers[0], labels=lbls)
    ax.set_xlabel('Heart Disease')
    plt.savefig('../img/dist_heart.png', bbox_inches='tight')

    #plotting the TSNE representation of Hearts dataset and saving it
    features = df_heart.loc[:, :'ST_Slope']
    tsne = TSNE(n_components=2, random_state=0)
    projections = tsne.fit_transform(features)
    fig = px.scatter(
        projections, x=0, y=1,
        color=df_heart.HeartDisease, labels={'color': 'Target'},
        color_continuous_scale=px.colors.sequential.Bluered
    )
    fig.update_layout(
        margin=dict(l=5, r=5, t=5, b=5),
    )
    fig.write_image("../img/tnse_heart.png")

    #spliting data into X and y for training
    X = df_heart.drop('HeartDisease', axis=1)
    y = df_heart['HeartDisease']

    return X, y

def process_data_multiclass():
    """
        Function to load, pre-process and separate into X and y the Thyroid data
        Return: 
            X: features that will be used in training 
            y: target feature 
    """
    #loading data
    data_train = pd.read_excel("data/ann-train.xlsx", header=None)
    data_test = pd.read_excel("data/ann-test.xlsx", header=None)
    #defining column names
    columns = ['age', 'sex','on_thyroxine','query_on_thyroxine','on_antithyroid_medication','sick', \
        'pregnant','thyroid_surgery','I131_treatment', \
        'query_hypothyroid','query_hyperthyroid', 'lithium', 'goitre', \
        'tumor','hypopituitary', 'psych', 'TSH', 'T3', 'TTT4', 'T4U', 'FTI', 'Target']
    data_train.columns = columns
    data_test.columns = columns
    #concatanating data
    data = pd.concat([data_train, data_test], ignore_index=True, sort=False)

    #spliting data into X and y for training
    X = data.drop('Target', axis=1)
    y = np.array(data['Target'])

    #under sampling data to have balenced classes
    under = RandomUnderSampler()
    X, y = under.fit_resample(X, y)

    #plotting the TSNE representation of Thyroid dataset and saving it
    tsne = TSNE(n_components=2, random_state=0)
    projections = tsne.fit_transform(X)
    fig = px.scatter(
        projections, x=0, y=1,
        color=y, labels={'color':'Target'}, 
        color_continuous_scale=px.colors.sequential.Agsunset
    )
    fig.update_layout(
    margin=dict(l=5, r=5, t=5, b=5),
    )
    fig.write_image("../img/tnse_tyroid.png")

    return X, y