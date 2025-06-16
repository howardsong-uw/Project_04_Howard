# spaceship_titanic_ensemble.py

from data_loader import load_data
from preprocess import preprocess_with_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np

def main():
    train_df, test_df, submission = load_data()
    train_df = preprocess_with_features(train_df)
    test_df = preprocess_with_features(test_df)

    features = [
        'HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side',
        'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
        'CabinNum', 'TotalSpend', 'GroupSize', 'IsAlone'
    ]

    X = train_df[features]
    y = train_df['Transported'].astype(int)
    X_test = test_df[features]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    lgbm = LGBMClassifier(random_state=42, n_estimators=100, learning_rate=0.1, num_leaves=15)
    rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=20)

    lgbm.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    val_preds_lgbm = lgbm.predict_proba(X_val)[:, 1]
    val_preds_rf = rf.predict_proba(X_val)[:, 1]

    val_preds_ensemble = (val_preds_lgbm + val_preds_rf) / 2
    val_preds_final = (val_preds_ensemble > 0.5).astype(int)
    val_acc = accuracy_score(y_val, val_preds_final)

    if __name__ == "__main__":
        print(f"Validation Accuracy (Ensemble): {val_acc:.4f}")

    test_preds_lgbm = lgbm.predict_proba(X_test)[:, 1]
    test_preds_rf = rf.predict_proba(X_test)[:, 1]
    test_preds_final = ((test_preds_lgbm + test_preds_rf) / 2) > 0.5

    submission['Transported'] = test_preds_final.astype(bool)
    submission.to_csv("submission_ensemble.csv", index=False)

    if __name__ == "__main__":
        print("submission_ensemble.csv saved.")

if __name__ == "__main__":
    main()
