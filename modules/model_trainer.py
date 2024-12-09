import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
import joblib
import os

class ModelTrainer:
    def __init__(self):
        self.historical_model = None
        self.news_model = None
        self.combined_model = None
        self.scaler = StandardScaler()
        self.models_dir = 'models'
        
        # Create models directory if it doesn't exist
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def train_historical_model(self, data):
        """
        Train model on historical price and technical indicators
        """
        # Prepare features
        feature_columns = ['open', 'high', 'low', 'close', 'tick_volume',
                         'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line',
                         'BB_upper', 'BB_lower', 'BB_middle', 'Volume_MA']
        
        X = data[feature_columns].values
        y = data['label'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        joblib.dump(self.scaler, f'{self.models_dir}/historical_scaler.pkl')
        
        # Reshape data for LSTM (samples, time steps, features)
        X_reshaped = self._prepare_sequences(X_scaled, seq_length=10)
        y_reshaped = y[10:]  # Remove first 10 points to match X_reshaped
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_reshaped, y_reshaped, test_size=0.2, shuffle=False
        )
        
        # Create and train model
        self.historical_model = self._create_lstm_model(
            input_shape=(X_reshaped.shape[1], X_reshaped.shape[2])
        )
        
        self.historical_model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        
        # Save model
        self.historical_model.save(f'{self.models_dir}/historical_model.h5')

    def train_news_model(self, news_data):
        """
        Train model on news sentiment data
        """
        # Prepare features
        X = news_data[['sentiment']].values
        y = news_data['label'].values  # Assuming we have labels for news impact
        
        # Scale features
        news_scaler = StandardScaler()
        X_scaled = news_scaler.fit_transform(X)
        joblib.dump(news_scaler, f'{self.models_dir}/news_scaler.pkl')
        
        # Create and train model
        self.news_model = Sequential([
            Dense(64, activation='relu', input_shape=(1,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(3, activation='softmax')  # 3 classes: buy, sell, hold
        ])
        
        self.news_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.news_model.fit(
            X_scaled, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        
        # Save model
        self.news_model.save(f'{self.models_dir}/news_model.h5')

    def train_combined_model(self, historical_data, news_data):
        """
        Train a combined model using both historical and news data
        """
        # Create input layers
        historical_input = Input(shape=(10, 14))  # 10 time steps, 14 features
        news_input = Input(shape=(1,))
        
        # Historical data branch
        x1 = LSTM(64, return_sequences=True)(historical_input)
        x1 = LSTM(32)(x1)
        x1 = Dense(16, activation='relu')(x1)
        
        # News data branch
        x2 = Dense(16, activation='relu')(news_input)
        x2 = Dense(8, activation='relu')(x2)
        
        # Combine branches
        combined = Concatenate()([x1, x2])
        combined = Dense(32, activation='relu')(combined)
        combined = Dropout(0.2)(combined)
        output = Dense(3, activation='softmax')(combined)
        
        # Create model
        self.combined_model = Model(
            inputs=[historical_input, news_input],
            outputs=output
        )
        
        self.combined_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Save model
        self.combined_model.save(f'{self.models_dir}/combined_model.h5')

    def _prepare_sequences(self, data, seq_length):
        """
        Prepare sequences for LSTM input
        """
        X = []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
        return np.array(X)

    def _create_lstm_model(self, input_shape):
        """
        Create LSTM model for historical data
        """
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(3, activation='softmax')  # 3 classes: buy, sell, hold
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def load_models(self):
        """
        Load trained models
        """
        try:
            self.historical_model = tf.keras.models.load_model(
                f'{self.models_dir}/historical_model.h5'
            )
            self.news_model = tf.keras.models.load_model(
                f'{self.models_dir}/news_model.h5'
            )
            self.combined_model = tf.keras.models.load_model(
                f'{self.models_dir}/combined_model.h5'
            )
            self.scaler = joblib.load(f'{self.models_dir}/historical_scaler.pkl')
            return True
        except:
            return False
