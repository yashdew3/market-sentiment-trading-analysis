import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import os

def load_and_prepare_data(path):
    df = pd.read_csv(path)
    df.dropna(inplace=True)

    features = [
        'Total_PnL', 'Total_Trade_Volume', 'Trade_Count',
        'PnL_3D_Mean', 'PnL_3D_STD',
        'Volume_3D_Mean', 'TradeCount_3D_Mean'
    ]
    X = df[features]
    y = df['Sentiment_Binary']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    return X_resampled, y_resampled, scaler

def train_best_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }

    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    print("✅ Best Parameters:", grid.best_params_)
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("✅ Classification Report:\n", classification_report(y_test, y_pred))

    return best_model

def save_model(model, scaler, path="outputs/models"):
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, os.path.join(path, "sentiment_rf_model_optimized.pkl"))
    joblib.dump(scaler, os.path.join(path, "scaler_optimized.pkl"))
    print("🧠 Model and scaler saved successfully.")

if __name__ == "__main__":
    data_path = "data/feature_engineered_data.csv"
    X, y, scaler = load_and_prepare_data(data_path)
    best_model = train_best_model(X, y)
    save_model(best_model, scaler)
