import logging
from abc import ABC, abstractclassmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract  class for all models
    """
    @abstractclassmethod
    def train(self, X_train, y_train):
        """
        trains in model
        """
        pass

class LinearRegressionModel(Model):
    """
    trains the model
    """
    def  train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        Args:
            X_train: training data
            y_train: training label
        Returns:
            None
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.erro("Error in cleaning data: {}".format(e))
            raise e
