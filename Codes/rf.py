import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

def load_and_preprocess_data(filepath):
    df = pd.read_excel(filepath)
    features = df.drop(['Pressure_Strain', 'FileType', 'Dataset','Lx'], axis=1)
    target = df['Pressure_Strain']

    # Handle NaNs
    features.fillna(features.mean(), inplace=True)
    target.fillna(target.mean(), inplace=True)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    dump(scaler, 'scaler_gb.joblib')  # Save the scaler
    return features, target

def train_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    mse = mean_squared_error(y_test, model.predict(X_test))
    print(f"Training Mean Squared Error: {mse}")
    dump(model, 'gradient_boosting_model.joblib')

def main():
    filepath = 'combined_data_all_reynolds_20PI_100PI.xlsx'
    features, target = load_and_preprocess_data(filepath)
    train_model(features, target)

if __name__ == "__main__":
    main()
