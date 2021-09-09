# Disable warnings
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import math
from math import sqrt
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression 
from pydataset import data

##########################################################################

def plot_residuals(y, yhat):
    '''
    This function takes in a dataframe, the y target variable 
    and the yhat (model predictions) and creates columns for residuals
    and baseline residuals. It returns a graph of both residual columns.
    '''

    # create a residual column
    df['residual'] = (yhat - y)

    # create a residual baseline column
    df['residual_baseline'] = (y.mean() - y)
    
    fig, ax = plt.subplots(figsize=(13,7))

    ax.hist(df.residual_baseline, label='baseline residuals', alpha=.6)
    ax.hist(df.residual, label='model residuals', alpha=.6)
    ax.legend()

    plt.show()

###############################################################################

def regression_errors(df, y, yhat):
    '''
    
    '''
    
    SSE = mean_squared_error(y, yhat)*len(df)


    MSE = mean_squared_error(y, yhat)


    RMSE = sqrt(mean_squared_error(y, yhat))


    ESS = sum((yhat - y.mean())**2)
    TSS = sum((y - y.mean())**2)

    # compute explained variance
    R2 = ESS / TSS
    
    print('SSE is:', SSE)
    print('ESS is:', ESS)
    print('TSS is:', TSS)
    print('R2 is:', R2)
    print('MSE is:', MSE)
    print('RMSE is:', RMSE)    

######################################################################


def baseline_mean_errors(df, y, yhat_baseline):
    
    SSE_baseline = mean_squared_error(y, yhat_baseline)*len(df)
    
    MSE_baseline = mean_squared_error(y, yhat_baseline)
    
    RMSE_baseline = sqrt(mean_squared_error(y, yhat_baseline))
    
    
    print('Baseline SSE is:', SSE_baseline)
    print('Baseline MSE is:', MSE_baseline)
    print('Baseline RMSE is:', RMSE_baseline)

#######################################################################


def better_than_baseline(df, y, yhat, yhat_baseline):

        RMSE = sqrt(mean_squared_error(y, yhat))
    
        RMSE_baseline = sqrt(mean_squared_error(y, yhat_baseline))
        
        if RMSE < RMSE_baseline:
            print('True - The model performs better than the baseline')
            
        elif RMSE > RMSE_baseline:
            print('False - The baseline performs better than the model')
        
        return RMSE, RMSE_baseline