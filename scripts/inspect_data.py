import pandas as pd

from src.data.load import TARGET, load_raw

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

df = load_raw()
print("Shape:", df.shape)
print("\n--- dtypes / non-null ---")
df.info()
print("\n--- describe ---")
print(df.describe().T)
print("\n--- missing values ---")
print(df.isna().sum())
print("\n--- target balance ---")
print(df[TARGET].value_counts(normalize=True))
print("\nDuplicated rows:", df.duplicated().sum())
print("\n--- categorical values ---")
for c in df.select_dtypes(include=["object", "string"]):
    print(c, df[c].value_counts().to_dict())