## Importing data:

import pandas as pd
from sklearn.metrics import accuracy_score
from MongoDB_Prep import DataCleaning
from Training_model import prep_new_data, prep_data_train, split_data_train, split_new_data, training_random_forest, prediction_random_forest, random_forest_feature_importance


def main():

    DataCleaning()
    
    ## Load Data
    # Load train data
    churn_train_orig = pd.read_csv('churn.csv')

    # Load new data
    churn_new_orig = pd.read_csv('churn_new_customers.csv')

    ## Prediction
    churn_train = prep_data_train(churn_train_orig)
    churn_new = prep_new_data(churn_new_orig)
    
    x_train, y_train, psn_train = split_data_train(churn_train)
    x_new, psn_new = split_new_data(churn_new)
    model = training_random_forest(12, 9, 1, x_train, y_train)
    churn_new_orig_with_predict = prediction_random_forest(model, x_new, churn_new_orig)
    random_forest_feature_importance(model, x_new)
    
    ## Accuracy
    y_train_pred_RandomForest = model.predict(x_train)
    test_acc = accuracy_score(y_train, y_train_pred_RandomForest)

    ## Saving new data prediction to csv
    churn_new_orig_with_predict['churn_predict'] = churn_new_orig_with_predict['churn_predict'].apply(lambda x: 'Yes' if x == 1 else 'No')
    churn_new_orig_with_predict.to_csv('churn_prediction.csv', index=False)

if __name__ == '__main__':
    main()