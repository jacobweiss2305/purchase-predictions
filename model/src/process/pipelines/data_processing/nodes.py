import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing

def stat_plots(df: pd.DataFrame, column_name: str):
    print(df[column_name].describe())
    plt.subplot(1, 2, 1)
    ax = df[column_name].hist()
    ax.set_title(f"Distribution of {column_name}")
    ax.set_xlabel(f"{column_name}")
    ax.set_ylabel("Count")

    plt.subplot(1, 2, 2)
    ax = df.boxplot(column=column_name)
    ax.set_title(f"Distribution of {column_name}")
    ax.set_ylabel("Count")                    

    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=1.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.4)

    plt.show()

def stacked_bar_chart(df: pd.DataFrame, legend: str):
    x = df.columns
    y2 = df.iloc[1]
    plt.bar(x, y2, color='b')
    plt.title(legend)
    plt.show()

def standardize(df):
    values=['total_clicks', 'seconds_per_session', 'count_unique_items']
    x = df[values]
    scaler = preprocessing.StandardScaler()
    scaler.fit(x)
    scaled = pd.DataFrame(scaler.transform(x), columns=values).fillna(0).set_index(df.index)
    return df.drop(values, axis = 1).join(scaled)