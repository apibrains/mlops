import logging
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluation our model
    """

    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the scores for the model

        Args:
            y_true: True labels
            y_pred: Predicted labels
        """

        pass

class MSE(Evaluation):
    """
    Evaluation Strategy that uses Mean Sequared Error
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        calculating Means Squared Error.

        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))

class R2(Evaluation):
    """
    Evaluation Strategy that uses R2 Score
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        calculating score using R2 score.

        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2 Score: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating R2 Score: {}".format(e))
            raise e

class RMSE(Evaluation):
    """
    Evaluation Strategy that uses Root Mean Squared Error
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE")
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info("RMSE: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in calculating RMSE: {}".format(e))
            raise e