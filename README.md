# GROUP53_CENG313 README

Kaggle Playground Series S5E1: Forecasting Sticker Sales
https://www.kaggle.com/competitions/playground-series-s5e1
 
## Team Members 
| Name | Student ID | GitHub Username |
|------|------------|-----------------| 
| Simge Altın | 23118080076 | simgealtin |

## Dataset
Kaggle Playground Series S5E1: Forecasting Sticker Sales. The task is to predict daily sticker sales (`num_sold`) for different countries, stores and products. The evaluation metric is MAPE.

- `train.csv`: 230,130 rows, 2010-01-01 to 2016-12-31
- `test.csv`: 98,550 rows, 2017-01-01 to 2019-12-31 (no `num_sold`)
- Columns: `id`, `date`, `country`, `store`, `product`, `num_sold`
- 6 countries x 3 stores x 5 products = 90 series
- `num_sold` has 8,871 missing values (3.9%), 99% of them in the Holographic Goose series of Canada and Kenya

## Approach
**EDA (Assignment 1):** missing value analysis, yearly/monthly/weekday patterns, breakdown by country, store and product.

Key findings:
- `num_sold` is right-skewed, so a `log1p` transform is used for the target.
- Kenya sells far less (mean ~20) than other countries (mean 540-1420); this causes the second peak in the log distribution.
- Weekday effect is strong (Sunday ~22% higher than Monday); monthly seasonality is weaker (~6%).
- There is no clear overall yearly trend.

**Validation:** time-based split (train on 2010-2015, validate on 2016), since the test set is in the future.

**Missing values:** rows with missing `num_sold` are dropped for the baselines.

**Features:** month, day of week, day of year, weekend flag, country, store, product.

**Baselines:** Naive (series mean), Ridge (one-hot features), LightGBM (categorical features). Ridge and LightGBM are trained on `log1p(num_sold)`.

Reusable code is in `src/` (`data.py`, `models.py`, `utils.py`) and used by the notebooks.

## Results
| Model | Validation MAPE (%) |
|-------|---------------------|
| Naive (series mean) | 18.78 |
| Ridge (log1p target) | 17.47 |
| LightGBM (log1p target) | 14.91 |

**Error analysis (LightGBM, validation):**

| Country | MAPE (%) |
|---------|----------|
| Norway | 28.45 |
| Kenya | 20.86 |
| Canada | 14.32 |
| Italy | 10.48 |
| Singapore | 8.07 |
| Finland | 7.62 |

The errors are concentrated in Norway and Kenya. Norway's yearly mean sales decrease after 2011 (1622 in 2011, 1131 in 2016). Since the baseline models do not use the year, they learn the average level of the training years and over-predict in 2016. This will be addressed in Assignment 2.

## How to Run
1. Download `train.csv`, `test.csv` and `sample_submission.csv` from the Kaggle competition's Data tab.
2. Place them in a folder (the `data/` folder in the repository root is git-ignored, or a Google Drive folder such as `CENG313_data/`).
3. Open `assignment1_eda_baseline.ipynb` in Google Colab.
4. Run the first cell (clones this repository and adds `src/` to the path), then mount Google Drive.
5. Set `DATA_DIR` in the notebook to the folder that contains the CSV files.
6. Run all cells in order (Runtime -> Run all).
7. Install the required libraries with `pip install -r requirements.txt` (Colab already has most of them).

## Progress Log
| Date | What |
|------|------|
| 2026-11-07 | Repo created, basic EDA (missing values, date ranges, country distribution) |
| 2026-11-08 | Time series EDA, product and store breakdown |
| 2026-11-10 | MAPE and time split, Naive/Ridge/LightGBM baselines, error analysis, first Kaggle submission |

## Kaggle Submission History
| Date | Method | Public Score | Private Score |
|------|--------|--------------|---------------|
| 2026-11-10 | LightGBM baseline (log1p target, trained on all of train) | 0.13275 | 0.15166 |
