import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

class Model(ABC):
    """
    Abstract class for all models
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Training the model

        Args:
            X_train: Training data
            y_train: Testing labels

        Retuns:
            None
        """

        pass


class LinearRegressionModel(Model):
    """
    Linear Regression Model
    """

    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        Args:
            X_train: Training data
            y_train: Training labels
        """
        
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error("Error in training model: {}".format(e))

class RandomForestModel(Model):
    """
    Random Forest Model
    """

    def train(self, X_train, y_train, **kwargs):
        """
        Trains the mode

        Args:
            X_train: Training Data
            y_train: Training labels
        """

        try:
            forest = RandomForestRegressor( max_depth=100, max_features=10,**kwargs)
            forest.fit(X_train, y_train)
            logging.info("Model training completed")
            return forest
        except Exception as e:
            logging.error("Error in training model {}".format(e))