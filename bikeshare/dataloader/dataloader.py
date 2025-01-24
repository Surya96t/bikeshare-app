import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

class DataLoader:
    """ Data loading and preprocessing class """
    
    @staticmethod
    def load_data(data_config):
        """ Load data from file """
        df = pd.read_csv(data_config.path)
        return df
    
    @staticmethod
    def create_dat_month_col(data_config, dataset=None):
        """ Create month column """
        dataset['date'] = pd.to_datetime(dataset['date'], dayfirst=True)
        dataset["day"] = dataset["date"].dt.day_name()
        dataset['month'] = dataset['date'].dt.month_name()
        return dataset
    
    @staticmethod
    def drop_rows_and_columns(data_config, dataset=None):
        """ Drop unwanted rows and columns """
        
        """ We drop dew_point_temp because it has a high correlation with temp """
        dataset = dataset.drop(["date", "functioning_day", "dew_point_temp"], axis=1)
        dataset = dataset[dataset["rented_bike_count"] != 0]
        
        return dataset
    
    @staticmethod
    def preprocess_data(data_config, dataset=None):
        """ Preprocess data """
        dataset = DataLoader.create_dat_month_col(data_config, dataset)
        dataset = DataLoader.drop_rows_and_columns(data_config, dataset)
        
        X = dataset[data_config.X]
        y = dataset[data_config.y]
        
        test_size = data_config.test_size
        random_state = data_config.random_state
        
        # Splitting the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        X_train_num = X_train.select_dtypes(include=[np.number]).columns
        X_train_cat = X_train.select_dtypes(exclude=[np.number]).columns
        
        # Normalizing the numerical features and endcoding the categorical features
        scaler = MinMaxScaler()
        encoder = OneHotEncoder()
        
        col_transformer = ColumnTransformer(
            [
                ("num", scaler, X_train_num),
                ("cat", encoder, X_train_cat)
            ]
        )
        
        X_train = col_transformer.fit_transform(X_train)
        X_test = col_transformer.transform(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        return (
            X, y, X_train, X_test, y_train, y_test, col_transformer
        )
        
        
    