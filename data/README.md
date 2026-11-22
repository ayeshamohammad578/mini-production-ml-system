# data/

Raw and split CSVs are **not committed** (they are regenerated deterministically).

`python -m src.data_pipeline` downloads the public *Give Me Some Credit* training file
(Kaggle, 2011; public mirror URL in `src/config.py`) into `data/cs-training.csv` and writes
`train.csv`, `val.csv` and `test.csv` (stratified 70/15/15, seed 42).
