# Spaceship Titanic: ML Classification Project

This project is a solution to the [Kaggle Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) competition. It uses a feature-engineered dataset combined with machine learning models including LightGBM and Random Forest.

## Project Structure

```
.
├── data/                         # Data folder (auto-downloaded)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── main.py                      # Training LGBM and generating submission
├── download_data.py             # Script to download and extract data using Kaggle API
├── submit.py                    # Submit submission_ensemble.csv
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # Project documentation

```

## Dependencies

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Kaggle API Setup

1. Login to [Kaggle](https://www.kaggle.com/account)
2. Create a new API token and download `kaggle.json`
3. Place it in:\
   **Windows:** `C:\Users\<YourUsername>\.kaggle\kaggle.json`\
   **Linux/Mac:** `~/.kaggle/kaggle.json`

## Download Data

To fetch and unzip the dataset automatically:

```bash
python download_data.py
```

## Train and Submit

Train the model and generate a submission file using LightGBM:

```bash
python main.py
```

## Features Used

- Categorical encoding of `HomePlanet`, `CryoSleep`, `Destination`, `VIP`, etc.
- Derived features:
  - `GroupSize` from passenger group ID
  - `IsAlone` indicating if traveling solo
  - `TotalSpend` on amenities
- Numerical columns normalized where needed

## Results

Model: `LightGBMClassifier + RandomForest`\
Validation Accuracy: \~80.5%


