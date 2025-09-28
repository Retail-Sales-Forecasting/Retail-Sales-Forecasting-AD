from pathlib import Path
import pandas as pd

DATA_DIR = Path('data')

def load_train():
    return pd.read_csv(DATA_DIR / 'train.csv', parse_dates=['date'], low_memory=False)

def load_holidays():
    return pd.read_csv(DATA_DIR / 'holidays_events.csv', parse_dates=['date'], low_memory=False)

def load_stores():
    return pd.read_csv(DATA_DIR / 'stores.csv', low_memory=False)
