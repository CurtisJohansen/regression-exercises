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


############################################################################
######################### Acquire Telco Function ###########################

#import libraries
import pandas as pd
import numpy as np
import os
from pydataset import data
from sklearn.model_selection import train_test_split

# acquire
import acquire


# Create helper function to get the necessary connection url.
def get_connection(db_name):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    from env import host, user, password
    return f'mysql+pymysql://{user}:{password}@{host}/{db_name}'


#create function to retrieve telco_churn data with specific columns
def get_telco_data():
    '''
    This function reads in the Telco Churn data from the Codeup db
    and returns a pandas DataFrame with customer_id, monthly_charges, tenure, total_charges columns
    for customers who have 2-year contracts.
    '''
    
    sql_query = '''
    SELECT customer_id, monthly_charges, tenure, total_charges
    FROM customers
    WHERE contract_type_id LIKE '3'
    '''
    return pd.read_sql(sql_query, get_connection('telco_churn'))

############################ ALL Telco Data Function ##############################

#create function to retrieve telco_churn data with all columns
def all_telco_data(df):
    '''
    This function reads in the Telco Churn data from the Codeup db
    and returns a pandas DataFrame with all columns
    '''
    
    sql_query = '''
    SELECT *
    FROM customers
    '''
    return pd.read_sql(sql_query, get_connection('telco_churn'))

############################ Wrangle Telco Function ##############################


def wrangle_telco():
    '''
    This function checks to see if telco_churn.csv already exists, 
    if it does not, one is created
    then the data is cleaned and the dataframe is returned
    '''
    #check to see if telco_churn.csv already exist
    if os.path.isfile('telco_churn.csv'):
        df = pd.read_csv('telco_churn.csv', index_col=0)
    
    else:

        #creates new csv if one does not already exist
        df = get_telco_data()
        df.to_csv('telco_churn.csv')

    #replace blank spaces and special characters
    df = df.replace(r'^\s*$', np.nan, regex=True)

    #change total_charges to float from object
    df.total_charges = df.total_charges.str.replace(' ', '0').astype(float)

    #fill NaN values (10 out of 1695) with monthly charges
    df.total_charges = df.total_charges.fillna(df.monthly_charges)

    return df

############################ Tenure Years Function ##############################

def months_to_years(df):
    '''
    this function accepts the telco churn dataframe
    and returns a dataframe with a new feature in complete years of tenure
    '''
    df['tenure_years'] = df.tenure / 12
    return df

###########################################################################
############################ Split Data Function ##############################

def split_data(df):
    '''
    split our data,
    takes in a pandas dataframe
    returns: three pandas dataframes, train, test, and validate
    '''
    #create train_validate and test datasets
    train, test = train_test_split(df, train_size = 0.8, random_state = 123)
    #create train and validate datasets
    train, validate = train_test_split(train, train_size = 0.7, random_state = 123)

    # Have function print datasets shape
    print(f'train -> {train.shape}')
    print(f'validate -> {validate.shape}')
    print(f'test -> {test.shape}')
   
    return train, validate, test
