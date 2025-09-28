import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent
FULL_PATH = DATA_DIR / 'train.csv'
SAMPLE_PATH = DATA_DIR / 'train_sample.csv'


def generate_sample(n_rows=1000, random_state=42):
    if not FULL_PATH.exists():
        raise FileNotFoundError(f"{FULL_PATH} not found. Please download it from Kaggle.")

    df = pd.read_csv(FULL_PATH, parse_dates=['date'], low_memory=False)
    sample = df.sample(n=n_rows, random_state=random_state)
    sample.to_csv(SAMPLE_PATH, index=False)
    print(f"✅ Sample saved to {SAMPLE_PATH} ({n_rows} rows)")


if __name__ == '__main__':
    generate_sample()
