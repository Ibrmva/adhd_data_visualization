
import pandas as pd
from pathlib import Path

DATA_FOLDER = Path("data")

files = [
    "<1_2019.csv",
    "2019-23.csv",
    "all ages_19-23.csv",
    "geo_dis_<1.csv",
    "geo_distribution.csv"
]

for filename in files:
    path = DATA_FOLDER / filename

    print("\n" + "=" * 75)
    print(f"FILE: {filename}")
    print("=" * 75)

    if not path.exists():
        print("FILE NOT FOUND")
        continue

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("\nColumn names:")
    print(df.columns.tolist())

    for column in [
        "location_name",
        "sex_name",
        "age_name",
        "metric_name",
        "measure_name",
        "cause_name",
        "year"
    ]:
        if column in df.columns:
            print(f"\n{column}:")
            print(sorted(df[column].dropna().astype(str).unique().tolist()))

    print("\nFirst 3 records:")
    print(df.head(3).to_string(index=False))
