## Importing data:

import pandas as pd
from pandas import json_normalize
import pymongo
from pymongo import MongoClient



def DataCleaning():

    client = MongoClient('mongodb://localhost:27017/')
    db = client['BDA']
    
    ## Data Cleaning:
    
    list_cursor = list(db.Customers.find({}))
    
    normalized_data = json_normalize(list_cursor)
    
    df = pd.DataFrame(normalized_data)
    
    df = df.drop(columns=['_id'])
    
    df.columns = [col.split(".")[-1] for col in df.columns]
    
    df['PaperlessBilling'] = df['PaperlessBilling'].str[0]
    
    df.TotalCharges = pd.to_numeric(df.TotalCharges,errors='coerce')
    
    df['MonthlyCharges'] = df['MonthlyCharges'].round(2)
    
    df['tenure'] = df['tenure'].round().astype(float)
    
    df = df.dropna()
    
    column_order = [
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", 
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", 
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", 
        "MonthlyCharges", "TotalCharges"
    ]
    df = df[column_order]
    
    df.to_csv('churn_new_customers.csv', index =False)
