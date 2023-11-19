import logging
import pandas as pd
from zenml import step

@step
def evaluate_model(df: pd.DataFrame) -> None:
    """
    Evaluattes the model on the ingested data
    Args:
        df: the ingested data
    """