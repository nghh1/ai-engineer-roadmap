import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured["total_tokens"] = featured["prompt_tokens"] + featured["completion_tokens"]
    featured["latency_seconds"] = featured["latency_ms"] / 1000
    valid_latency = featured["latency_seconds"] > 0
    featured.loc[valid_latency, "tokens_per_second"] = (
        featured.loc[valid_latency, "total_tokens"] / featured.loc[valid_latency, "latency_seconds"]
    )
    featured.loc[featured["latency_seconds"].isna(), "tokens_per_second"] = np.nan
    return featured