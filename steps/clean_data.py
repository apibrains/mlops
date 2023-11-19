import logging
import pandas as pd
from zenml import step

@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning the ingested data

    Args:
        df: ingested data
    """
    pass