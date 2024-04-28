import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import datetime

def load_and_preprocess_data(filepath):
    df = pd.read_excel(filepath)
    features = df.drop(['Pressure_Strain', 'Pressure_Transport', 'FileType', 'Dataset', 'Lx'], axis=1)
    target = df['Pressure_Strain']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pd.to_pickle(scaler, 'scaler.pkl')
    return X_train, X_test, y_train, y_test

def build_and_compile_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(128),
        BatchNormalization(),
        Activation('relu'),
        Dense(128),
        Dropout(0.3),
        Activation('relu'),
        BatchNormalization(),
        Dense(64),
        Activation('relu'),
        Dense(64),
        Dropout(0.3),
        Activation('relu'),
        BatchNormalization(),
        Dense(32),
        Activation('relu'),
        Dense(1, activation='linear')
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model

def train_model(X_train, y_train, X_val, y_val):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    model = build_and_compile_model(X_train.shape[1])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    history = model.fit(X_train, y_train, epochs=200, validation_data=(X_val, y_val),
                        callbacks=[early_stopping, model_checkpoint, reduce_lr, tensorboard_callback], batch_size=32, verbose=1)
    return model, history

def main():
    filepath = 'training.xlsx'
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)
    model, history = train_model(X_train, y_train, X_test, y_test)
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')
    model.save('final_model.keras')  # Save the final model after training

if __name__ == "__main__":
    main()
