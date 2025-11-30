# src/data_loader.py

import os
import pandas as pd

def load_data():
    """
    Loads and merges:
    - train.csv
    - test.csv
    - stores.csv
    - holidays_events.csv
    - transactions.csv

    Returns:
        train_df, test_df
    """

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")

    print("DATA DIR:", data_dir)

    # -------------------------
    # Load raw CSV files
    # -------------------------
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df  = pd.read_csv(os.path.join(data_dir, "test.csv"))
    stores_df = pd.read_csv(os.path.join(data_dir, "stores.csv"))
    holidays_df = pd.read_csv(os.path.join(data_dir, "holidays_events.csv"))
    transactions_df = pd.read_csv(os.path.join(data_dir, "transactions.csv"))

    # -------------------------
    # Convert dates to datetime
    # -------------------------
    train_df["date"] = pd.to_datetime(train_df["date"])
    test_df["date"]  = pd.to_datetime(test_df["date"])
    holidays_df["date"] = pd.to_datetime(holidays_df["date"])
    transactions_df["date"] = pd.to_datetime(transactions_df["date"])

    # -------------------------
    # Rename columns to avoid conflicts
    # -------------------------
    # stores.csv has a column named "type"
    stores_df = stores_df.rename(columns={"type": "store_type"})

    # holidays_events.csv has a column named "type"
    holidays_df = holidays_df.rename(columns={"type": "holiday_type"})

    # -------------------------
    # Merge stores into train/test
    # -------------------------
    train_df = train_df.merge(stores_df, on="store_nbr", how="left")
    test_df  = test_df.merge(stores_df, on="store_nbr", how="left")

    # -------------------------
    # Merge holiday info
    # -------------------------
    train_df = train_df.merge(
        holidays_df[["date", "holiday_type"]],
        on="date", how="left"
    )
    test_df  = test_df.merge(
        holidays_df[["date", "holiday_type"]],
        on="date", how="left"
    )

    # -------------------------
    # Merge transactions
    # -------------------------
    train_df = train_df.merge(
        transactions_df, on=["date", "store_nbr"], how="left"
    )
    test_df = test_df.merge(
        transactions_df, on=["date", "store_nbr"], how="left"
    )

    # -------------------------
    # Create is_holiday column
    # -------------------------
    HOLIDAY_TYPES = ["Holiday", "Additional", "Transferred"]

    train_df["is_holiday"] = train_df["holiday_type"].apply(
        lambda x: 1 if x in HOLIDAY_TYPES else 0
    )
    test_df["is_holiday"] = test_df["holiday_type"].apply(
        lambda x: 1 if x in HOLIDAY_TYPES else 0
    )

    # -------------------------
    # Fill missing values for transactions
    # -------------------------
    train_df["transactions"] = train_df["transactions"].fillna(0)
    test_df["transactions"] = test_df["transactions"].fillna(0)

    # -------------------------
    # Final fillna for safety
    # -------------------------
    train_df.fillna(0, inplace=True)
    test_df.fillna(0, inplace=True)

    return train_df, test_df
