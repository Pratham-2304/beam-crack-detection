from pathlib import Path
import pandas as pd
from ml.inference import predict_from_excel

# folder with damaged excel files
DATA_DIR = Path("damaged")

results = []

files = list(DATA_DIR.glob("*.xlsx"))[:10]  # test first 10 files

for f in files:
    try:
        pred = predict_from_excel(f)
        pred["file"] = f.name
        results.append(pred)
        print(f"OK → {f.name}")
    except Exception as e:
        print(f"FAIL → {f.name}: {e}")

df = pd.DataFrame(results)
df.to_csv("baseline_predictions.csv", index=False)

print("\nSaved baseline_predictions.csv")
print(df)
