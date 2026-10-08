#!/usr/bin/env bash
# End-to-end pipeline. Run from the repository root.
set -e
[ -f data/online_retail_II.csv ] || python scripts/download_data.py
python src/preprocess.py
python src/eda.py
python src/train_als.py
python src/recommend.py 12347
