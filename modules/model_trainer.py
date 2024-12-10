import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import joblib
import os

class ModelTrainer:
    def __init__(self, risk_level='medium'):
        self.models = {
            'historical': None,
            'news': None,
            'combined': None,
            'regime_specific': {}  # Models for different market regimes
        }
        self.scaler = StandardScaler()
        self.models_dir = 'models'
        
        # More aggressive risk levels
        self.risk_levels = {
            'low': 0.45,      # More aggressive
            'medium': 0.35,   # Much more aggressive
            'high': 0.25,     # Very aggressive
            'dynamic': None   # Will be set based on market conditions
        }
        self.risk_level = risk_level
        
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def train_historical_model(self, data, model_dir, symbol):
        """Train enhanced historical model with regime awareness"""
        # Prepare features
        feature_columns = [
            'open', 'high', 'low', 'close', 'tick_volume',
            'Returns', 'Momentum_1h', 'Momentum_15m', 'ATR',
            'Volatility', 'Volume_Ratio', 'Channel_Position',
            'RSI', 'RSI_Divergence', 'ADX'
        ]
        
        if symbol == "BTCUSD":
            feature_columns.extend(['Price_Range', 'Volume_Price_Trend', 'Buying_Pressure'])
        
        X = data[feature_columns].values
        y = data['label'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        joblib.dump(self.scaler, os.path.join(model_dir, 'historical_scaler.pkl'))
        
        # Prepare sequences - shorter for more dynamic trading
        seq_length = 6  # 30-minute sequences for 5-min data
        X_seq = self._prepare_sequences(X_scaled, seq_length)
        y_seq = y[seq_length:]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, shuffle=False
        )
        
        # Calculate class weights to handle hold bias
        class_weights = self._calculate_class_weights(y_train)
        
        # Create and train model
        self.models['historical'] = self._create_lstm_model(
            input_shape=(X_seq.shape[1], X_seq.shape[2]),
            symbol=symbol
        )
        
        # Train with enhanced callbacks
        callbacks = self._get_training_callbacks(model_dir, 'historical')
        
        self.models['historical'].fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            class_weight=class_weights,
            callbacks=callbacks
        )
        
        # Train regime-specific models
        for regime in ['normal', 'volatile', 'trending']:
            regime_mask = data['Volatility_Regime' if regime == 'volatile' 
                             else 'Trend_Regime'].isin([regime])
            if regime_mask.sum() > 100:  # Only if enough data
                X_regime = X_scaled[regime_mask[seq_length:]]
                y_regime = y[regime_mask[seq_length:]]
                
                if len(X_regime) > seq_length:
                    X_regime_seq = self._prepare_sequences(X_regime, seq_length)
                    y_regime_seq = y_regime[seq_length:]
                    
                    regime_model = self._create_lstm_model(
                        input_shape=(seq_length, X_scaled.shape[1]),
                        symbol=symbol,
                        regime=regime
                    )
                    
                    regime_model.fit(
                        X_regime_seq, y_regime_seq,
                        epochs=50,
                        batch_size=16,
                        validation_split=0.2,
                        class_weight=class_weights,
                        callbacks=self._get_training_callbacks(model_dir, f'historical_{regime}')
                    )
                    
                    self.models['regime_specific'][regime] = regime_model
        
        # Save models
        self.models['historical'].save(os.path.join(model_dir, 'historical_model.keras'))
        for regime, model in self.models['regime_specific'].items():
            model.save(os.path.join(model_dir, f'historical_{regime}_model.keras'))

    def train_news_model(self, news_data, model_dir):
        """Train enhanced news model with temporal awareness"""
        # Create features from sentiment and metadata
        features = ['sentiment', 'credibility', 'time_weight']
        X = news_data[features].values
        
        # Create labels with reduced hold bias
        news_data['label'] = 1  # Default to hold
        news_data.loc[news_data['sentiment'] > 0.15, 'label'] = 2  # Lower threshold for long
        news_data.loc[news_data['sentiment'] < -0.15, 'label'] = 0  # Lower threshold for short
        y = news_data['label'].values
        
        # Scale features
        news_scaler = StandardScaler()
        X_scaled = news_scaler.fit_transform(X)
        joblib.dump(news_scaler, os.path.join(model_dir, 'news_scaler.pkl'))
        
        # Create enhanced news model
        self.models['news'] = Sequential([
            Dense(64, activation='relu', input_shape=(len(features),)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')
        ])
        
        self.models['news'].compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Calculate class weights
        class_weights = self._calculate_class_weights(y)
        
        # Train with callbacks
        callbacks = self._get_training_callbacks(model_dir, 'news')
        
        if len(X_scaled) > 100:
            self.models['news'].fit(
                X_scaled, y,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                class_weight=class_weights,
                callbacks=callbacks
            )
        else:
            self.models['news'].fit(
                X_scaled, y,
                epochs=50,
                batch_size=8,
                validation_split=0.2,
                class_weight=class_weights,
                callbacks=callbacks
            )
        
        # Save model
        self.models['news'].save(os.path.join(model_dir, 'news_model.keras'))

    def train_combined_model(self, historical_data, news_data, model_dir, symbol):
        """Train enhanced combined model with market regime awareness"""
        # Create input layers
        historical_input = Input(shape=(6, 15))  # 6 time steps, 15 features
        news_input = Input(shape=(3,))  # sentiment, credibility, time_weight
        
        # Historical data branch with residual connections
        x1 = LSTM(64, return_sequences=True)(historical_input)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.3)(x1)
        x1 = LSTM(32)(x1)
        x1 = BatchNormalization()(x1)
        
        # News data branch
        x2 = Dense(32, activation='relu')(news_input)
        x2 = BatchNormalization()(x2)
        x2 = Dense(16, activation='relu')(x2)
        
        # Combine branches
        combined = Concatenate()([x1, x2])
        combined = Dense(64, activation='relu')(combined)
        combined = BatchNormalization()(combined)
        combined = Dropout(0.3)(combined)
        combined = Dense(32, activation='relu')(combined)
        combined = BatchNormalization()(combined)
        output = Dense(3, activation='softmax')(combined)
        
        # Create model
        self.models['combined'] = Model(
            inputs=[historical_input, news_input],
            outputs=output
        )
        
        # Use higher learning rate for more dynamic adaptation
        self.models['combined'].compile(
            optimizer=Adam(learning_rate=0.002),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Save model
        self.models['combined'].save(os.path.join(model_dir, 'combined_model.keras'))

    def _prepare_sequences(self, data, seq_length):
        """Prepare sequences for LSTM input"""
        X = []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
        return np.array(X)

    def _create_lstm_model(self, input_shape, symbol, regime=None):
        """Create enhanced LSTM model with batch normalization"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(64),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(3, activation='softmax')
        ])
        
        # Higher learning rate for BTC and volatile regimes
        lr = 0.002 if symbol == "BTCUSD" or regime == 'volatile' else 0.001
        
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def _calculate_class_weights(self, y):
        """Calculate balanced class weights with reduced hold bias"""
        classes = np.unique(y)
        weights = dict(zip(classes, [1.0] * len(classes)))
        
        # Count samples per class
        class_counts = np.bincount(y)
        total_samples = len(y)
        
        # Calculate balanced weights
        for cls in classes:
            weights[cls] = total_samples / (len(classes) * class_counts[cls])
            
        # Reduce weight for hold class
        if 1 in weights:  # 1 is the hold class
            weights[1] *= 0.7  # Reduce hold weight by 30%
            
        return weights

    def _get_training_callbacks(self, model_dir, model_name):
        """Get callbacks for training"""
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                os.path.join(model_dir, f'{model_name}_best.keras'),
                monitor='val_loss',
                save_best_only=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            )
        ]

    def get_confidence_threshold(self):
        """Get confidence threshold based on risk level"""
        return self.risk_levels.get(self.risk_level, 0.35)  # Default to medium if invalid

    def set_risk_level(self, risk_level):
        """Set risk level for trading"""
        if risk_level in self.risk_levels:
            self.risk_level = risk_level
            return True
        return False

    def load_models(self, symbol):
        """Load all trained models for a symbol"""
        try:
            model_dir = f'models/{symbol.lower()}'
            self.models['historical'] = tf.keras.models.load_model(
                os.path.join(model_dir, 'historical_model.keras')
            )
            self.models['news'] = tf.keras.models.load_model(
                os.path.join(model_dir, 'news_model.keras')
            )
            self.models['combined'] = tf.keras.models.load_model(
                os.path.join(model_dir, 'combined_model.keras')
            )
            
            # Load regime-specific models if they exist
            for regime in ['normal', 'volatile', 'trending']:
                regime_path = os.path.join(model_dir, f'historical_{regime}_model.keras')
                if os.path.exists(regime_path):
                    self.models['regime_specific'][regime] = tf.keras.models.load_model(regime_path)
            
            self.scaler = joblib.load(os.path.join(model_dir, 'historical_scaler.pkl'))
            return True
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False
