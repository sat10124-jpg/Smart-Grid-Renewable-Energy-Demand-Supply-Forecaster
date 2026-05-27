# Importing tools 
import pandas as pd 
import numpy as np
# import XGBoost and sklearn's train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def engineer_features(cleaned_df):
    # loading the cleaned dataset
    df = cleaned_df.copy() # Deep copy so that the orignla datfrom is not modified
    #creates an column called "hour" from the index
    df['hour'] = df.index.hour
    # creating an column called "day_of_week" from the index
    df['day_of_week'] = df.index.dayofweek
    # creating an column called "month" from the index
    df['month'] = df.index.month

    # creating lag features for the solar generation column (lags of 1, 2,
    # and 3 hours) and adding them to the dataframe
    df['lag_1'] = df['DE_solar_generation_actual'].shift(1)
    df['lag_24'] = df['DE_solar_generation_actual'].shift(24)
    df['lag_48'] = df['DE_solar_generation_actual'].shift(48)
    # drop any rows that now have NaN
    df.dropna(inplace=True)

    return df

def train_model(df):
    # taking the engineered dataframe as input  
    # and defining X as all columns except solar, y as solar generationn column only 
    # split X and y into train/tests sets (80% train, 20% test)
    # with shuffle=False since this is time series data
    X = df.drop(columns=['DE_solar_generation_actual'])
    y = df['DE_solar_generation_actual']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    # train an XGBoost model on the training data
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    return model, X_test, y_test




if __name__ == "__main__":
    # import and call pipeline fucntions from data_pipeline.py
    from data_pipeline import download_data, load_and_clean
    download_data()
    cleaned_df = load_and_clean()
    # call engineer_features on the cleaned data and call train_model and print "Model trained successfully"
    engineered_df = engineer_features(cleaned_df)
    model, X_test, y_test = train_model(engineered_df)
    print("Model trained successfully")
