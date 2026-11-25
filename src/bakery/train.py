import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    data = pd.read_csv("output/bakery/dataset.csv")
    
    features = data.drop(columns=["label"])
    labels = data["label"]

   
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(features, labels)

    joblib.dump(model, "output/bakery/model.pkl")
