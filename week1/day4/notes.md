# Components in a DataFrame
Index, label, row, column

Each column in a DataFrame is a Series

# Create a Series/DataFrame
pd.Series([1, 2, 3], name="column_name")
pd.DataFrame({
    "A": 1.0,
    "B": pd.date_range("20260904", periods=4),
    "C": pd.Series(2, index=list(range(4)), dtype="float32"),
    "D": np.array([3] * 4, dtype="int32"),
    "E": pd.Categorical(["train", "test", "train", "test"]),
    "F": "valid"
})

# Operation on a DataFrame
## Basic information about a DataFrame
df.info()
## Count no. elements in a column, its mean, std, and 5 summary stats
df.describe() 
## Display first/last n rows of a DataFrame
df.head(n)
df.tail(n)
## Display indexes/labels
df.index
df.columns
## Filter rows from a DataFrame via boolean indexing
df[df["column"] < 3]
df[df["column"].isin([1, 2])]
df[df["column"].notna()]
## Get specific columns/rows from a DataFrame via label-/position-based selection
df.loc[df["column"] < 3, "column_name"]
df.iloc[:3, :2]
## Both .loc and .iloc can be used to adjust the selected data
df.iloc[:2, 4] = "unknown"
## DataFrame to numpy array
df.to_numpy()
## DataFrame sorts by an axis or values
df.sort_index(axis=1)
df.sort_values(by="B")

# Working with missing data
## Return a boolean object indicating whether values are NA
df.isna()
df.isna().sum()
## Drop rows/columns that have missing data
df.dropna(axis=0)
df.dropna(axis=1)
## Replace missing values with non-NA data
df.fillna(0)
## Fill gaps forward/backward (use last/next available value)
df.ffill()
df.bfill()

# Operations on unique/duplicated data
## Detect duplicated rows
df.duplicated()
df.drop_duplicates()
## Return unique values of a column (record first appearance)
df["column"].unique()
## Frequency of each distinct values/rows
df["column"].value_counts()
df.value_counts()

## String normalisation
df["column"] = df["column"].str.strip()
## A column contains numbers may unexpectedly be object type
pd.to_numeric(df["column"], errors='coerce')

# Aggregating statistics
df.agg({
    "colA": ["min", "max", "mean", "skew"],
    "colB": ["min", "max", "mean", "median"]
})
## Aggregating statistics grouped by category
df[["Category", "Feature"]].groupby("Category").mean()
df.groupby("Category")["Feature"].mean()
## Aggregating statistics grouped by multiple categories
df.groupby(["CategoryA", "CategoryB"])["Feature"].mean()
## Aggregating function apply to each numerical column
df.groupby("Category").mean(numeric_only=True)