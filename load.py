import pandas as pd
from pandas import DataFrame

# Funkce pro načítání dat podle cesty
def load_data(path: str) -> DataFrame:
    df = pd.read_csv(path)
    return df