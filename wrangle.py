# Disable warnings
import warnings
warnings.filterwarnings("ignore")

# Import libraries 
import pandas as pd
import numpy as np
import os

# Library dealing with NA values

from sklearn.impute import SimpleImputer

# Import to obtain data from Codeup SQL databse server

from env import host, user, password


# connection url

# This function uses my info from the env file to create a connection url to access the Codeup db.
# It takes in a string name of a database as an argument.

def get_connection(db, user=user, host=host, password=password):
    
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


# acquiring data for the first time

def get_zillow_db():
    '''
    This function reads in the zillow data from the Codeup db
    and returns a pandas DataFrame with all columns and it was joined with other tables.
    '''
    sql_query = '''
        select bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet,
         taxvaluedollarcnt, yearbuilt, taxamount, fips from properties_2017
        join propertylandusetype
        on propertylandusetype.propertylandusetypeid = properties_2017.propertylandusetypeid
        and propertylandusetype.propertylandusetypeid = 261
        '''
    return pd.read_sql(sql_query, get_connection('zillow'))


# main function to acquire project data

def get_zillow():
    '''
    This function reads in zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = get_zillow_db()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow.csv')
        
    return df

# Replace the null values with the mean value of each column

def impute_null_values():
    '''
    Using SimpleImputer to impute the mean value into the null values into each column.
    '''
    #Using the mean imputer function

    imputer = SimpleImputer(strategy='mean')

    # Create a for loop that will impute all the null values in each one of our columns.

    for col in df.columns:
        df[[col]] = imputer.fit_transform(df[[col]])
    return df


# Function to remove outliers in dataframe

  def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df  

# Function that combines the above functions

    def wrangle_zillow():
    '''
    We will call the other functions from the file in our wrangle_zillow function.
    '''
    #Acquire data
    df = get_zillow()

    #Clean data
    df = impute_null_values(df)

    #Remove outliers
    df = remove_outliers(df, 2.5, df.columns)

    #Return DataFrame
    return df