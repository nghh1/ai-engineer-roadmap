import pandas as pd

def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    cleaned = raw_df.copy()
    cleaned["model"] = cleaned["model"].str.lower().str.strip()
    cleaned["request_id"] = pd.to_numeric(cleaned["request_id"], errors="coerce")
    for nc in ["prompt_tokens", "completion_tokens", "latency_ms"]:
        cleaned[nc] = pd.to_numeric(cleaned[nc], errors="coerce")
        cleaned[nc] = cleaned[nc].fillna(cleaned[nc].median())
    cleaned.drop_duplicates(inplace=True)
    cleaned.reset_index(drop=True, inplace=True)
    return cleaned
