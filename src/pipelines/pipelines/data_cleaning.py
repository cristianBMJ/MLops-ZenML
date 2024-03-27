import logging 
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract class defininf strategy for handling data
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStrategy( DataStrategy):
    """
    Strategy for preprocessing data
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data = data.drop([
                "order_aproved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimeted_delivery_date",
                "order_purchase_timestamp"
                ], axis=1 )
            data["product_weight_g"].fillna( data["product_weight_g"].median() ,  inplace=True) 
            data["product_lenght_g"].fillna( data["product_lenght_g"].median() ,  inplace=True) 
            data["product_heidht_g"].fillna( data["product_height_g"].median() ,  inplace=True) 
            data["product_width_g"].fillna( data["product_width_g"].median() ,  inplace=True) 
            data["review_comment_message"].fillna( "No review",  inplace=True) 

            data = data.select_dtypes( include = [np.number])
            cols_to_drop = ["custimer_zip_code_prefix", "order_time_id"]
            data = data.drop( cols_to_drop,axis=1)
            return data
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e) )
            raise e

class DataDivideStrategy( DataStrategy):
    """
    Strategy for dividing into train and test
    """
    def handle_data(self, data: pd.DataFrame) -> Union[ pd.DataFrame , pd.Series]:
        try:
            X = data.drop( ["review_score"], axis=1)
            y = data["review_score"]
            X_train, x_test, y_train, y_test = train_test_split( X, y , test_size=0.2, random_state=42)
            return X_train, x_test, y_train, y_test 
        except Exception as e:
            logging.error( "Error in dividing data: {e}".format(e))
            raise
        
class DataCleaninig:
    """
    Class for cleaning data which processes the data and divides it into train and test
    """ 
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data  
        self.strategy = strategy

    def handle_data(self) -> Union[ pd.DataFrame, pd.Series]:
        """
        Handle data
        """
        try:
            return self.strategy.handle_data( self.data)
        except Exception as e:
            logging.error( "Error in handling Data: {}".format(e) )
            raise e
        
if __name__ == "__main__":
    data = pd.read_csv("/workspaces/MLops-ZenML/data/olist_customers_dataset.csv")
    data_cleaning = DataCleaninig(data, DataPreProcessStrategy)
    data_cleaning.handle_data()

