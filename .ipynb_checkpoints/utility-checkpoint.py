# Importing Libraries
from IPython.display import display
import pandas as pd
import numpy as np

# Loading the Data
def load_data():
    campaign_dataset = pd.read_csv('data/campaign_data.csv')
    coupon_item_mapping = pd.read_csv('data/coupon_item_mapping.csv')
    customer_demographics = pd.read_csv('data/customer_demographics.csv')
    customer_transaction_data = pd.read_csv('data/customer_transaction_data.csv')
    item_data = pd.read_csv('data/item_data.csv')
    train_dataset = pd.read_csv('data/train.csv')
    return campaign_dataset, coupon_item_mapping, customer_demographics, customer_transaction_data, item_data, train_dataset

# Merging all the data into 1 data
def merge_data(campaign_dataset, coupon_item_mapping, customer_demographics, customer_transaction_data, item_data, train_dataset):
    cust_demo_data=pd.merge(train_dataset,customer_demographics,on='customer_id')
    campaign_data=pd.merge(train_dataset,campaign_dataset,on='campaign_id')
    data=pd.merge(train_dataset,customer_demographics,on='customer_id')
    coupon_item_data=pd.merge(coupon_item_mapping,item_data ,on='item_id')
    coupon_item_data.groupby('coupon_id')['category'].value_counts()
    customer_transaction_data =pd.merge(customer_transaction_data ,item_data ,on='item_id')
    data=pd.merge(cust_demo_data,campaign_data)
    data=pd.merge(data,coupon_item_data)
    data=pd.merge(data,customer_transaction_data)
    return data

# Proprocessing all the data
def preprocess_data(data):
    data=data.drop(['id'], axis=1)
    data=data.drop(['date','start_date','end_date'],axis=1)
    data.marital_status.fillna("Married", inplace = True) 
    data['no_of_children'].fillna("1", inplace = True) 
    data['age_range'].fillna(data['age_range'].mode()[0], inplace=True)
    data['rented'].fillna(data['rented'].mode()[0], inplace=True)
    data['family_size'].fillna(data['family_size'].mode()[0], inplace=True)
    data['income_bracket'].fillna(data['income_bracket'].mode()[0], inplace=True)
    return data
    
# Making two dataset X (Model Input) y(target) for the model
def X_y(data):
    X = data.drop('redemption_status',1)
    X = pd.get_dummies(X)
    y = data.redemption_status
    return X, y

# Seeing all the datasets
def data_vis(campaign_dataset, coupon_item_mapping, customer_demographics, customer_transaction_data, item_data, train_dataset):
    print('Train')
    display(train_dataset.head())
    print('--'*50)
    print('campaign_dataset')
    display(campaign_dataset.head())
    
    print('--'*50)
    print('coupon_item_mapping')
    display(coupon_item_mapping.head())
    
    print('--'*50)
    print('customer_demographics')
    display(customer_demographics.head())
    
    print('--'*50)
    print('customer_transaction_data')
    display(customer_transaction_data.head())
    
    print('--'*50)
    print('item_data')
    display(item_data.head())

# Checking in all dataset if there is any NaN values
def any_nan_val(campaign_dataset, coupon_item_mapping, customer_demographics, customer_transaction_data, item_data, train_dataset):
    print('Train')
    display(train_dataset.isnull().any())
    print('--'*50)
    print('campaign_dataset')
    display(campaign_dataset.isnull().any())
    
    print('--'*50)
    print('coupon_item_mapping')
    display(coupon_item_mapping.isnull().any())
    
    print('--'*50)
    print('customer_demographics')
    display(customer_demographics.isnull().any())
    
    print('--'*50)
    print('customer_transaction_data')
    display(customer_transaction_data.isnull().any())
    
    print('--'*50)
    print('item_data')
    display(item_data.isnull().any())