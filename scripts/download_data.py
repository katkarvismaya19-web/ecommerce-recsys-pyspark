"""Download UCI Online Retail II and convert both yearly sheets into one CSV at data/online_retail_II.csv.

Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii  (CC BY 4.0)
"""
import io
import os
import urllib.request
import zipfile

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
os.makedirs("data", exist_ok=True)
print("Downloading", URL)
raw = urllib.request.urlopen(URL).read()
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    name = [n for n in z.namelist() if n.endswith(".xlsx")][0]
    sheets = pd.read_excel(z.open(name), sheet_name=None, dtype={"Invoice": str, "StockCode": str})
df = pd.concat(sheets.values(), ignore_index=True)
df.to_csv("data/online_retail_II.csv", index=False)
print(f"Saved {len(df):,} rows to data/online_retail_II.csv")
