
import os
import pandas as pd

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")

    # Load CSVs
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    stores_df = pd.read_csv(os.path.join(data_dir, "stores.csv"))
    holidays_df = pd.read_csv(os.path.join(data_dir, "holidays_events.csv"))

    # Merge
    train_df = train_df.merge(stores_df, on="store_nbr", how="left")
    train_df = train_df.merge(holidays_df, on="date", how="left")
    test_df = test_df.merge(stores_df, on="store_nbr", how="left")
    test_df = test_df.merge(holidays_df, on="date", how="left")

    # Holiday flag
    train_df["is_holiday"] = train_df["type_y"].apply(lambda x: 1 if x=="Holiday" else 0)
    test_df["is_holiday"] = test_df["type_y"].apply(lambda x: 1 if x=="Holiday" else 0)

    # Fill missing
    train_df.fillna(0, inplace=True)
    test_df.fillna(0, inplace=True)

    return train_df, test_df
