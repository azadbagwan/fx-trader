import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
import joblib
import time
import threading
import logging

class TradingBot:
    def __init__(self):
        self.is_running = False
        self.thread = None
        self.historical_model = None
        self.news_model = None
        self.combined_model = None
        self.scaler = None
        self.current_positions = {}
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0
        }
        
        # Setup logging
        logging.basicConfig(
            filename='trading_bot.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        self._load_models()

    def _load_models(self):
        """
        Load the trained models and scaler
        """
        try:
            self.historical_model = tf.keras.models.load_model('models/historical_model.h5')
            self.news_model = tf.keras.models.load_model('models/news_model.h5')
            self.combined_model = tf.keras.models.load_model('models/combined_model.h5')
            self.scaler = joblib.load('models/historical_scaler.pkl')
            logging.info("Models loaded successfully")
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            raise Exception("Failed to load models")

    def start(self, symbol, lot_size=0.1, stop_loss_pips=50, take_profit_pips=100):
        """
        Start the trading bot
        """
        if not mt5.initialize():
            logging.error("MetaTrader5 initialization failed")
            raise Exception("MetaTrader5 initialization failed")

        self.symbol = symbol
        self.lot_size = lot_size
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
        
        self.is_running = True
        self.thread = threading.Thread(target=self._trading_loop)
        self.thread.start()
        
        logging.info(f"Trading bot started for {symbol}")

    def stop(self):
        """
        Stop the trading bot
        """
        self.is_running = False
        if self.thread:
            self.thread.join()
        self._close_all_positions()
        logging.info("Trading bot stopped")

    def _trading_loop(self):
        """
        Main trading loop
        """
        while self.is_running:
            try:
                # Get current market data
                current_data = self._get_market_data()
                
                # Get predictions from both models
                historical_pred = self._get_historical_prediction(current_data)
                news_pred = self._get_news_prediction()
                
                # Combine predictions and make trading decision
                decision = self._make_trading_decision(historical_pred, news_pred)
                
                # Execute trades based on decision
                self._execute_trades(decision)
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Wait for next iteration
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logging.error(f"Error in trading loop: {str(e)}")
                time.sleep(60)  # Wait before retrying

    def _get_market_data(self):
        """
        Get current market data and prepare it for prediction
        """
        # Get last 100 candles to ensure we have enough data for features
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M15, 0, 100)
        
        if rates is None:
            raise Exception("Failed to get market data")
        
        # Convert to DataFrame and calculate technical indicators
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Calculate technical indicators (same as in DataProcessor)
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['close'].rolling(window=20).std()
        
        # Volume indicators
        df['Volume_MA'] = df['tick_volume'].rolling(window=20).mean()
        
        return df

    def _get_historical_prediction(self, data):
        """
        Get prediction from historical data model
        """
        feature_columns = ['open', 'high', 'low', 'close', 'tick_volume',
                         'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line',
                         'BB_upper', 'BB_lower', 'BB_middle', 'Volume_MA']
        
        # Prepare features
        features = data[feature_columns].values
        features_scaled = self.scaler.transform(features)
        
        # Create sequence
        sequence = features_scaled[-10:].reshape(1, 10, -1)
        
        # Get prediction
        prediction = self.historical_model.predict(sequence)
        return prediction[0]

    def _get_news_prediction(self):
        """
        Get prediction from news model
        """
        # This would typically involve getting recent news and processing it
        # For now, return neutral prediction if news data is not available
        return np.array([0.33, 0.34, 0.33])

    def _make_trading_decision(self, historical_pred, news_pred):
        """
        Combine predictions and make final trading decision
        """
        # Weighted average of predictions (giving more weight to historical data)
        combined_pred = 0.7 * historical_pred + 0.3 * news_pred
        
        # Get the highest probability action
        action = np.argmax(combined_pred)
        probability = combined_pred[action]
        
        # Only trade if probability is high enough
        if probability > 0.6:
            return action  # 0: Sell, 1: Hold, 2: Buy
        return 1  # Hold

    def _execute_trades(self, decision):
        """
        Execute trades based on decision
        """
        if decision == 0:  # Sell
            self._open_position("SELL")
        elif decision == 2:  # Buy
            self._open_position("BUY")

    def _open_position(self, trade_type):
        """
        Open a new trading position
        """
        point = mt5.symbol_info(self.symbol).point
        price = mt5.symbol_info_tick(self.symbol).ask
        
        if trade_type == "BUY":
            sl = price - self.stop_loss_pips * point
            tp = price + self.take_profit_pips * point
            type_ = mt5.ORDER_TYPE_BUY
        else:
            sl = price + self.stop_loss_pips * point
            tp = price - self.take_profit_pips * point
            type_ = mt5.ORDER_TYPE_SELL
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.lot_size,
            "type": type_,
            "price": price,
            "sl": sl,
            "tp": tp,
            "magic": 234000,
            "comment": "python script open",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Order failed: {result.comment}")
        else:
            logging.info(f"Order successful: {trade_type} {self.symbol}")

    def _close_all_positions(self):
        """
        Close all open positions
        """
        positions = mt5.positions_get(symbol=self.symbol)
        if positions:
            for position in positions:
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": position.volume,
                    "type": mt5.ORDER_TYPE_BUY if position.type == 1 else mt5.ORDER_TYPE_SELL,
                    "position": position.ticket,
                    "price": mt5.symbol_info_tick(self.symbol).bid,
                    "magic": 234000,
                    "comment": "python script close",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                result = mt5.order_send(close_request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logging.error(f"Failed to close position: {result.comment}")

    def _update_performance_metrics(self):
        """
        Update trading performance metrics
        """
        positions = mt5.positions_get(symbol=self.symbol)
        if positions:
            for position in positions:
                if position.ticket not in self.current_positions:
                    self.current_positions[position.ticket] = position.price_open
                    self.performance_metrics['total_trades'] += 1
                
                profit = position.profit
                if profit > 0:
                    self.performance_metrics['winning_trades'] += 1
                else:
                    self.performance_metrics['losing_trades'] += 1
                
                self.performance_metrics['total_profit'] += profit
                
                # Update max drawdown
                if profit < self.performance_metrics['max_drawdown']:
                    self.performance_metrics['max_drawdown'] = profit

    def get_performance_metrics(self):
        """
        Get current performance metrics
        """
        return self.performance_metrics
