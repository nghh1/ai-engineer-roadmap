import pandas as pd

def model_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("model").agg(
        requests = pd.NamedAgg("model", "count"),
        success_rate=pd.NamedAgg("success", "mean"),
        avg_latency_ms=pd.NamedAgg("latency_ms", "mean"),
        avg_total_tokens=pd.NamedAgg("total_tokens", "mean"),
        avg_tokens_per_second=pd.NamedAgg("tokens_per_second", "mean")
    )


