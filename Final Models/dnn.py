import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def load_and_preprocess_data(filepath):
    df = pd.read_excel(filepath)
    features = df.drop(['Pressure_Strain', 'FileType', 'Dataset','Lx'], axis=1)
    target = df['Pressure_Strain']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save the scaler for later use
    pd.to_pickle(scaler, 'scaler.pkl')

    return X_train, X_test, y_train, y_test, features.columns.tolist()

def build_and_compile_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def main():
    filepath = 'combined_data_all_reynolds_20PI_100PI.xlsx'
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(filepath)
    model = build_and_compile_model(len(feature_names))
    model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=1)
    test_loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}')
    model.save('dnn.h5')

if __name__ == "__main__":
    main()
