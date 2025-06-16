import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_with_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw Spaceship Titanic data and generate features:
      - Split Cabin into Deck, CabinNum, Side
      - Derive GroupSize and IsAlone from PassengerId
      - Fill missing Age and spending columns
      - Encode categorical columns
      - Compute TotalSpend
    Returns processed DataFrame.
    """
    df = df.copy()
    # Cabin split
    cabin = df['Cabin'].fillna('U/0/U').str.split('/', expand=True)
    df['Deck'], df['CabinNum'], df['Side'] = cabin[0], cabin[1], cabin[2]
    # Convert CabinNum to numeric
    df['CabinNum'] = pd.to_numeric(df['CabinNum'], errors='coerce').fillna(0)

    # Group features
    df['Group'] = df['PassengerId'].str.split('_').str[0]
    df['GroupSize'] = df.groupby('Group')['Group'].transform('count')
    df['IsAlone'] = (df['GroupSize'] == 1).astype(int)

    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        df[col] = df[col].fillna(0)

    # Encode categorical features
    for col in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']:
        df[col] = df[col].fillna('Unknown').astype(str)
        df[col] = LabelEncoder().fit_transform(df[col])

    # Derive TotalSpend
    df['TotalSpend'] = (
        df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
    )
    return df
