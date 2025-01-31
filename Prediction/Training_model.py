## Importing data:

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier



## Creating Functions:

### prep_new_data function
def prep_new_data(df):
    
    df = df.rename(columns=str.lower)
    
    df.totalcharges = pd.to_numeric(df.totalcharges,errors='coerce')
    df = df.dropna(subset=['totalcharges'])
    
    services = ['phoneservice', 'multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies']
    df['num_services'] = (df[services] != 'No').sum(axis=1)

    
    l = []
    
    for n,v in df.items():
        if df[n].dtypes == 'object':
            if set(df[n].unique()) == {'Yes', 'No'}: 
                l.append(n)

    
    for n,v in df.items():
        if df[n].dtypes == 'int64':
            df[n] = df[n].astype('float')
            df[n] = df[n].round(2)
            
    df['tenure'] = df['tenure'].round()
    
    for i in l:
        df[i] = (df[i] == 'Yes').astype('float')
        
    df = df.applymap(lambda x: 'No' if isinstance(x, str) and x.startswith('No ') else x)
    
    df_aside = df['customerid']
    df = df.drop('customerid', axis=1)
    
    df = pd.get_dummies(df)
    
    for n,v in df.items():
        if df[n].dtypes == 'bool': 
            df[n] = df[n].astype('float')

    df['customerid'] = df_aside
    
    print('')
    
    null = df.isna().sum().sum()
    print (f'There are {null} null values')
    
    dtype = df.dtypes.unique()[0]
    print (f'There are {dtype} dtype')
    
    print('')
    
    return df


### prep_data_train function
def prep_data_train (df_churn):

    df_churn = df_churn.rename(columns=str.lower)
    
    df_churn.totalcharges = pd.to_numeric(df_churn.totalcharges,errors='coerce')
    df_churn = df_churn.dropna(subset=['totalcharges'])
    
    services = ['phoneservice', 'multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies']
    df_churn['num_services'] = (df_churn[services] != 'No').sum(axis=1)

    
    l = []
    
    for n,v in df_churn.items():
        if df_churn[n].dtypes == 'object':
            if set(df_churn[n].unique()) == {'Yes', 'No'}: 
                l.append(n)

    
    for n,v in df_churn.items():
        if df_churn[n].dtypes == 'int64':
            df_churn[n] = df_churn[n].astype('float')
    
    for i in l:
        df_churn[i] = (df_churn[i] == 'Yes').astype('float')
        
    df_churn = df_churn.applymap(lambda x: 'No' if isinstance(x, str) and x.startswith('No ') else x)
    
    df_churn_aside = df_churn['customerid']
    df_churn = df_churn.drop(['customerid'], axis=1)
    
    df_churn = pd.get_dummies(df_churn)
    
    df_churn = df_churn.join(df_churn_aside)
    
    for n,v in df_churn.items():
        if df_churn[n].dtypes == 'bool': 
            df_churn[n] = df_churn[n].astype('float')

    
    print('')
    
    null = df_churn.isna().sum().sum()
    print (f'There are {null} null values')
    
    dtype = df_churn.dtypes.unique()[0]
    print (f'There are {dtype} dtype')
    
    print('')
    
    return df_churn


### split_data_train
def split_data_train(df):
    label = 'churn'
    psn = 'customerid'

    x_train = df.drop(label, axis=1)
    x_train = x_train.drop(psn, axis=1)
    y_train = df[label]
    psn_train = df[psn]
    
    return x_train, y_train, psn_train


### split_new_data
def split_new_data(df):
    psn = 'customerid'

    x_new = df.drop(psn, axis=1)
    psn_new = df[psn]
    
    return x_new, psn_new


### training_random_forest
def training_random_forest(n, m, r, x_train, y_train):

    model = RandomForestClassifier(n_estimators=n, max_depth=m, random_state=r)
    model.fit(x_train, y_train)
    
    return model


### rediction_random_forest
def prediction_random_forest(model, x_new, df_orig):

    y_new = model.predict(x_new) 
    y_new = pd.Series(y_new,name='churn_predict')
    output = df_orig.join(y_new)
    
    return output


### random_forest_feature_importance
def random_forest_feature_importance(model, x_new):

    feature_importances = model.feature_importances_ 
    features = x_new.columns
    stats = pd.DataFrame({'feature':features, 'importance':feature_importances})
    print(stats.sort_values('importance', ascending=False))

    stats_sort = stats.sort_values('importance', ascending=True)
    stats_sort.plot(y='importance', x='feature', kind='barh')
    plt.title('Feature Importance of Random Forest')
    plt.show()