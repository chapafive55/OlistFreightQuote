# Olist Freight Quote Model

Gavin Chapa. Code appendix for *Predicting Freight for a Quote Engine*.

Predicts the freight charge (R$) for a delivered, single-seller Olist order, then compares
Ridge regression (Model A) with gradient boosting (Model B) on a 70/15/15 time-based split.

## Files

| File | What it does |
|---|---|
| `freight_pipeline.py` | Loads the CSVs into SQLite, runs the SQL, splits the data by date, tunes and tests both models, and prints every number in the report |
| `freight_pull.sql` | The data pull: one row per delivered, single-seller order (94,376 rows) |
| `results/run_output.txt` | Output from a full run |
| `results/metrics.json` | The same results as JSON (rewritten each time the script runs) |
| `requirements.txt` | Python packages |
| `data/` | The six Olist CSVs the script uses, zipped |

## How to run

1. The data is already in `data/`: the six Olist CSVs, zipped to get under GitHub's upload limit.
   The script reads the zip files directly, so there's nothing to unzip.
   (Source: Olist, Brazilian E-Commerce Public Dataset, https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce, CC BY-NC-SA 4.0)
2. Run:

```bash
pip install -r requirements.txt
python freight_pipeline.py                 # reads ./data
python freight_pipeline.py --data <folder> # or point at another folder
```

The full run takes about 30 seconds. Results are fixed (seed 42, no early stopping), so
every run prints the same numbers.
