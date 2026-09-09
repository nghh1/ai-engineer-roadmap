import pandas as pd
from pathlib import Path
def load_data(path: Path | str) -> pd.DataFrame: 
    return pd.read_csv(path)
