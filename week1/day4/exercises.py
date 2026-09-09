import pandas as pd

data = {
    "model": [
        "gpt", "claude", "gpt", "gemini",
        "claude", "gpt", "gemini"
    ],
    "accuracy": [
        0.91, 0.88, 0.94, 0.90,
        0.92, 0.87, 0.93
    ],
    "latency_ms": [
        120, 180, 110, 150,
        170, 130, 140
    ],
    "requests": [
        1000, 800, 1200, 950,
        1100, 700, 1050
    ]
}
df = pd.DataFrame(data)

# Exercise 1: inspection
"""
df.shape = (7, 4)
df["accuracy"].shape = (7, )
df[["accuracy"]].shape = (7, 1)
"""
print(df.head(3))
print(df.columns)
print(df.dtypes)
print(df.describe())

# Exercise 2: selection
selected_df = df[["model", "accuracy"]]
high_accuracy = df[df["accuracy"]>=0.90]
high_accuracy_models = df.loc[df["accuracy"]>=0.90, ["model", "accuracy"]]
print(high_accuracy_models)

# Exercise 3: multiple conditions
accuracy_and_latency = df[(df["accuracy"]>=0.90) & (df["latency_ms"]<150)]
accuracy_or_requests = df[(df["accuracy"]>=0.90) | (df["requests"]>=1100)]
print(accuracy_and_latency)
print(accuracy_or_requests)

# Exercise 4: sorting
print(df.sort_values(by=["accuracy", "latency_ms"], ascending=[False, True]))

# Exercise 5: aggregation
aggregation = df.agg({
    "accuracy": ["mean"],
    "latency_ms": ["mean", "max"],
    "requests": ["sum"]
})
aggregation_groupby_model = df.groupby("model").agg({
    "accuracy": ["mean"],
    "latency_ms": ["mean"],
    "requests": ["sum"]
})
print(aggregation)
print(aggregation_groupby_model)

# Exercise 6: dirty data
dirty_data = {
    "model": [
        " GPT ", "gpt", "Claude", "claude",
        "Gemini", "gemini", "gpt", "GPT "
    ],
    "accuracy": [
        "0.91", "0.94", "unknown", "0.92",
        "0.89", "0.93", "0.87", "0.91"
    ],
    "latency_ms": [
        120, 110, 180, None,
        150, 140, 130, 120
    ],
    "requests": [
        1000, 1200, 800, 1100,
        None, 1050, 700, 1000
    ]
}
dirty_df = pd.DataFrame(dirty_data)
dirty_df["model"] = dirty_df["model"].str.lower().str.strip()
dirty_df["accuracy"] = pd.to_numeric(dirty_df["accuracy"], errors='coerce')
dirty_df["accuracy"] = dirty_df["accuracy"].fillna(dirty_df["accuracy"].mean())
dirty_df["latency_ms"] = dirty_df["latency_ms"].fillna(dirty_df["latency_ms"].median())
dirty_df = dirty_df.dropna(axis=0, subset=["requests"])
clean_df = dirty_df.drop_duplicates().copy()
print(clean_df.info())
print(clean_df.isna().sum())
print(clean_df["model"].value_counts())

# Exercise 7: feature engineering
clean_df["latency_seconds"] = clean_df["latency_ms"] / 1000
clean_df["high_accuracy"] = (clean_df["accuracy"] >= 0.90)
clean_df["requests_per_ms"] = clean_df["requests"] / clean_df["latency_ms"]
print(clean_df)
