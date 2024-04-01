import logging

import mlflow
import pandas as pd 
from zenml import step

from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

from zenml.client import Client

experimenter_tracker = Client().active_stack.experiment_tracker

@step( experiment_tracker= experimenter_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig,
 ) -> RegressorMixin:
    """
    Trains the model on the ingested data

    Args:
        df: the ingested data  
    """
    try:
        model = None
        if config.model_name == "LinearRegression":
            model = LinearRegressionModel()
        
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError("Model {} not supported".format(config.model_name) )
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")