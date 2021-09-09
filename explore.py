import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

import env
import wrangle

########################## Tenure Function ##########################

def months_to_years(df):
    '''
    this function accepts the telco churn dataframe
    and returns a dataframe with a new feature in complete years of tenure
    '''
    df['tenure_years'] = (df.tenure/12).astype(int)
    return df


################### Plot Variable Pairs Function ##################

def plot_variable_pairs(df, target):
    sns.pairplot(df, corner = True, kind= 'reg', plot_kws={'line_kws':{'color':'red'}})


################### Plot Cat&Cont Function ##################
def plot_categorical_and_continuous_vars(df, categorical_var, continuous_var):
    '''
    this function that accepts your dataframe 
    and the name of the columns that hold the continuous and categorical features 
    outputs: 3 different plots for visualizing a categorical variable and a continuous variable.
    '''
    sns.swarmplot(data=df, x=categorical_var, y=continuous_var)
    plt.show()
    sns.boxplot(data=df, x=categorical_var, y=continuous_var)
    plt.show()
    sns.barplot(data=df, x=categorical_var, y=continuous_var)
    plt.show()