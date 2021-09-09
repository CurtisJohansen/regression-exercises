######################### Acquire Telco Function ###########################

#import libraries
import pandas as pd
import numpy as np
import os
from pydataset import data

# acquire
from env import host, user, password


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
    
    telco_query = '''
    SELECT customer_id, monthly_charges, tenure, total_charges
    FROM customers
    WHERE contract_type_id LIKE '3'
    '''
    return pd.read_sql(telco_query, get_connection('telco_churn'))

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
    

############################ Acquire Zillow Function ##############################

def get_zillow_data():
    '''
    This function reads in the Zillow data from the Codeup db
    and returns a pandas DataFrame with cbedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, and fips
    for all Single Family Residential properties.
    '''
    
    zillow_query = '''
    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
    FROM properties_2017
    WHERE propertylandusetypeid = 261
    '''
    return pd.read_sql(zillow_query, get_connection('zillow'))

############################ Acquire Mall Customers Function ##############################

def get_mall_customers_data():
    '''
    This function reads in the mall_customers data from the Codeup db
    and returns a pandas Dataframe
    '''
    
    mall_customers_query = '''
    SELECT *
    FROM customers
    '''
    return pd.read_sql(mall_customers_query, get_connection('mall_customers'))