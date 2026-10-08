# Data

The raw dataset is **not committed** (95 MB). Get it either way:

1. `python scripts/download_data.py` (downloads from UCI and writes `data/online_retail_II.csv`), or
2. Download manually from https://archive.ics.uci.edu/dataset/502/online+retail+ii, open the `.xlsx`, and save both sheets ("Year 2009-2010" and "Year 2010-2011") stacked into one CSV named `online_retail_II.csv` in this folder.

Licence: CC BY 4.0. Citation: Chen, D. Online Retail II [Dataset]. UCI Machine Learning Repository.

After running the pipeline this folder also holds `clean.parquet` and `interactions.parquet` (git-ignored).
