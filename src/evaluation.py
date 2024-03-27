import logging
from abc import ABC, abstractclassmethod

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error 

class Evaluation(ABC):
    """
    abstrac class defining strategy  for evaluation our models
    """
    @abstractclassmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """"
        Calculates the score of  the model
        Args:
            y_true: True labels
            y_pred: Predicted labels
        Returns:
            None
        """
        pass

class MSE(Evaluation):
    """Evaluation strategy that uses Mean Square Error"""
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error(f"Error while ingesting data: {e}")

class R2(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 score")
            r2 = r2_score(y_true, y_pred)
            logging.info("MSE: {}".format(r2))
            return r2
        except Exception as e:
            logging.error(f"Error while ingesting data: {e}")
    


class RMSE(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE")
            rmse =root_mean_squared_error(y_true, y_pred)
            logging.info("MSE: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error(f"Error while ingesting data: {e}")
