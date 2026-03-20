import pandas as pd
from pathlib import Path

RAW_DIR = Path(".")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

fake = pd.read_csv(RAW_DIR / "Fake.csv")
true = pd.read_csv(RAW_DIR / "True.csv")

fake["label"] = "FAKE"
true["label"] = "REAL"

df = pd.concat([
    fake[["text", "label"]],
    true[["text", "label"]]
])

df.to_csv(DATA_DIR / "fake_news.csv", index=False)

print("Saved data/fake_news.csv")
print(df["label"].value_counts())
