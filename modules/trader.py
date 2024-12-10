import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
import joblib
import time
import threading
import logging
import sys
import pytz
import os

class TradingBot:
    def __init__(self, risk_level='medium'):
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
        
        # Current state tracking
        self.current_state = {
            'position': 'None',
            'prediction': None,
            'timestamp': None,
            'last_price': None,
            'decision_confidence': 0.0,
            'risk_level': risk_level,
            'position_entry_time': None,
            'recent_trades': [],  # Store recent trade outcomes for adaptive learning
            'market_regime': 'normal',  # Track market regime (normal, volatile, trending)
            'volatility': 0.0  # Current market volatility
        }
        
        # Dynamic risk levels that adapt based on performance
        self.risk_levels = {
            'low': 0.45,     # More aggressive
            'medium': 0.35,  # Much more aggressive
            'high': 0.25,    # Very aggressive
            'dynamic': None  # Will be set based on market conditions
        }
        
        # Trading parameters - will be dynamically adjusted
        self.trading_params = {
            'BTCUSD': {
                'base_position_size': 0.01,
                'min_price_movement': 25,  # Reduced for more frequent trading
                'stop_loss_pips': 200,    # Tighter stops
                'take_profit_pips': 400,  # Balanced risk:reward
                'max_position_duration': timedelta(hours=4)  # Shorter holding time
            },
            'EURUSD': {
                'base_position_size': 0.1,
                'min_price_movement': 0.0002,
                'stop_loss_pips': 30,
                'take_profit_pips': 60,
                'max_position_duration': timedelta(hours=2)
            }
        }
        
        # Setup logging
        self.logger = logging.getLogger('TradingBot')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        
        fh = logging.FileHandler('trading_bot.log')
        ch = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _update_market_regime(self, data):
        """Dynamically assess market conditions"""
        recent_volatility = data['close'].pct_change().std() * np.sqrt(96)  # Annualized
        self.current_state['volatility'] = recent_volatility
        
        # Detect market regime
        returns = data['close'].pct_change()
        if recent_volatility > data['close'].pct_change().std() * 2:
            self.current_state['market_regime'] = 'volatile'
        elif abs(returns.mean() * np.sqrt(96)) > recent_volatility:
            self.current_state['market_regime'] = 'trending'
        else:
            self.current_state['market_regime'] = 'normal'
            
        # Adjust risk threshold based on market regime
        if self.current_state['market_regime'] == 'volatile':
            self.risk_levels['dynamic'] = 0.3  # More conservative in volatile markets
        elif self.current_state['market_regime'] == 'trending':
            self.risk_levels['dynamic'] = 0.25  # More aggressive in trending markets
        else:
            self.risk_levels['dynamic'] = 0.35  # Balanced in normal markets

    def _calculate_position_size(self):
        """Dynamic position sizing based on volatility and recent performance"""
        base_size = self.trading_params[self.symbol]['base_position_size']
        
        # Adjust for volatility
        vol_factor = 1.0
        if self.current_state['volatility'] > 0:
            vol_factor = 1.0 / self.current_state['volatility']
        
        # Adjust for recent performance
        if len(self.current_state['recent_trades']) >= 5:
            win_rate = sum(1 for t in self.current_state['recent_trades'][-5:] if t > 0) / 5
            perf_factor = 0.5 + win_rate  # Scale from 0.5 to 1.5 based on win rate
        else:
            perf_factor = 1.0
        
        return base_size * vol_factor * perf_factor

    def _get_market_data(self):
        """Get current market data with enhanced short-term indicators"""
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, 100)
            
            if rates is None:
                raise Exception(f"Failed to get market data for {self.symbol}")
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Short-term indicators
            df['SMA_5'] = df['close'].rolling(window=5).mean()
            df['SMA_10'] = df['close'].rolling(window=10).mean()
            df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
            
            # Momentum indicators
            df['RSI'] = self._calculate_rsi(df['close'], 14)
            df['MFI'] = self._calculate_mfi(df, 14)
            
            # Volatility and momentum
            df['ATR'] = self._calculate_atr(df, 14)
            df['Momentum'] = df['close'] - df['close'].shift(5)
            
            # Volume analysis
            df['Volume_MA'] = df['tick_volume'].rolling(window=5).mean()
            df['Volume_Spike'] = df['tick_volume'] > df['Volume_MA'] * 2
            
            return df.dropna()
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {str(e)}")
            raise

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_mfi(self, df, period=14):
        """Calculate Money Flow Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['tick_volume']
        
        delta = typical_price.diff()
        positive_flow = (money_flow.where(delta > 0, 0)).rolling(window=period).sum()
        negative_flow = (money_flow.where(delta < 0, 0)).rolling(window=period).sum()
        
        money_ratio = positive_flow / negative_flow
        return 100 - (100 / (1 + money_ratio))

    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(window=period).mean()

    def _make_trading_decision(self, historical_pred, news_pred):
        """Enhanced trading decision with momentum and adaptive thresholds"""
        # Get base predictions
        combined_pred = 0.7 * historical_pred + 0.3 * news_pred
        
        # Get market data for additional signals
        market_data = self._get_market_data()
        
        # Calculate momentum signals
        momentum_signal = 0
        if market_data['Momentum'].iloc[-1] > 0 and market_data['RSI'].iloc[-1] < 70:
            momentum_signal = 1  # Bullish
        elif market_data['Momentum'].iloc[-1] < 0 and market_data['RSI'].iloc[-1] > 30:
            momentum_signal = -1  # Bearish
            
        # Volume signal
        volume_signal = 1 if market_data['Volume_Spike'].iloc[-1] else 0
        
        # Adjust predictions based on technical signals
        if momentum_signal == 1:
            combined_pred[2] *= 1.2  # Boost long probability
            combined_pred[0] *= 0.8  # Reduce short probability
        elif momentum_signal == -1:
            combined_pred[0] *= 1.2  # Boost short probability
            combined_pred[2] *= 0.8  # Reduce long probability
            
        if volume_signal:
            # Boost the strongest signal on volume spikes
            max_idx = np.argmax(combined_pred)
            combined_pred[max_idx] *= 1.2
            
        # Normalize probabilities
        combined_pred = combined_pred / combined_pred.sum()
        
        # Get action and confidence
        action = np.argmax(combined_pred)
        confidence = combined_pred[action]
        
        # Store confidence
        self.current_state['decision_confidence'] = float(confidence)
        
        # Use dynamic threshold based on market regime
        threshold = self.risk_levels.get('dynamic', self.risk_levels[self.current_state['risk_level']])
        
        # If confidence exceeds threshold and we have confirming signals, return the action
        if confidence > threshold and (
            (action == 2 and momentum_signal == 1) or  # Long with bullish momentum
            (action == 0 and momentum_signal == -1)    # Short with bearish momentum
        ):
            return action
            
        return 1  # Hold if no strong signals

    def _execute_trades(self, decision):
        """Execute trades with dynamic position sizing and adaptive stops"""
        try:
            point = mt5.symbol_info(self.symbol).point
            price = mt5.symbol_info_tick(self.symbol).ask
            
            # Calculate dynamic position size
            position_size = self._calculate_position_size()
            
            # Calculate dynamic stops based on ATR
            market_data = self._get_market_data()
            atr = market_data['ATR'].iloc[-1]
            
            if self.symbol == "BTCUSD":
                stop_loss = atr * 2
                take_profit = atr * 4
            else:
                stop_loss = atr * 3
                take_profit = atr * 6
            
            if decision == 0:  # Short
                sl = price + stop_loss
                tp = price - take_profit
                trade_type = mt5.ORDER_TYPE_SELL
            elif decision == 2:  # Long
                sl = price - stop_loss
                tp = price + take_profit
                trade_type = mt5.ORDER_TYPE_BUY
            else:
                return
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": position_size,
                "type": trade_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "magic": 234000,
                "comment": f"regime:{self.current_state['market_regime']}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            self.logger.info(f"Sending trade request: {request}")
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Order failed: {result.comment}")
                raise Exception(f"Order failed: {result.comment}")
            
            self.logger.info(f"Order successful: {'Buy' if trade_type == mt5.ORDER_TYPE_BUY else 'Sell'} {self.symbol}")
            self.current_state['position_entry_time'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            raise

    def _trading_loop(self):
        """Enhanced trading loop with adaptive behavior"""
        while self.is_running:
            try:
                # Get current market data
                current_data = self._get_market_data()
                current_price = current_data['close'].iloc[-1]
                
                # Update market regime and risk parameters
                self._update_market_regime(current_data)
                
                # Update current state
                self.current_state['last_price'] = current_price
                self.current_state['timestamp'] = datetime.now()
                
                # Check and manage existing positions
                self._manage_positions()
                
                # Only get new predictions if no position is open
                if not self._has_open_position():
                    # Get predictions
                    historical_pred = self._get_historical_prediction(current_data)
                    news_pred = self._get_news_prediction()
                    
                    self.current_state['prediction'] = {
                        'historical': historical_pred.tolist(),
                        'news': news_pred.tolist()
                    }
                    
                    # Make trading decision
                    decision = self._make_trading_decision(historical_pred, news_pred)
                    confidence = self.current_state['decision_confidence']
                    
                    decision_map = {0: 'Short', 1: 'Hold', 2: 'Long'}
                    self.current_state['position'] = decision_map[decision]
                    
                    self.logger.info(f"Market Regime: {self.current_state['market_regime']}")
                    self.logger.info(f"Volatility: {self.current_state['volatility']:.4f}")
                    self.logger.info(f"Decision: {decision_map[decision]} (Confidence: {confidence:.2%})")
                    
                    # Execute trade if conditions are met
                    if decision != 1:  # If not holding
                        self._execute_trades(decision)
                
                # Shorter sleep time for more responsive trading
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(60)

    def _manage_positions(self):
        """Enhanced position management with trailing stops"""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions:
            for position in positions:
                # Calculate current profit in points
                current_price = mt5.symbol_info_tick(self.symbol).bid
                points = (current_price - position.price_open) * (1 if position.type == 0 else -1)
                
                # Update trailing stop if in profit
                if points > 0:
                    new_sl = None
                    if position.type == 0:  # Buy
                        new_sl = current_price - (points * 0.5)  # Move stop to lock in 50% of profits
                        if new_sl > position.sl:
                            self._modify_position(position.ticket, sl=new_sl)
                    else:  # Sell
                        new_sl = current_price + (points * 0.5)
                        if new_sl < position.sl:
                            self._modify_position(position.ticket, sl=new_sl)
                
                # Check for exit conditions
                if self._should_exit_trade(position):
                    self._close_position(position.ticket)
                    
                    # Update recent trades list
                    self.current_state['recent_trades'].append(position.profit)
                    if len(self.current_state['recent_trades']) > 20:
                        self.current_state['recent_trades'].pop(0)

    def _should_exit_trade(self, position):
        """Determine if we should exit a trade based on market conditions"""
        current_data = self._get_market_data()
        
        # Exit if momentum reverses
        if position.type == 0:  # Long position
            if current_data['Momentum'].iloc[-1] < 0 and current_data['RSI'].iloc[-1] > 70:
                return True
        else:  # Short position
            if current_data['Momentum'].iloc[-1] > 0 and current_data['RSI'].iloc[-1] < 30:
                return True
        
        # Exit if volume spike occurs against position
        if current_data['Volume_Spike'].iloc[-1]:
            if position.type == 0 and current_data['close'].iloc[-1] < current_data['close'].iloc[-2]:
                return True
            if position.type == 1 and current_data['close'].iloc[-1] > current_data['close'].iloc[-2]:
                return True
        
        return False

    def _modify_position(self, ticket, sl=None, tp=None):
        """Modify an existing position's stops"""
        request = {
            "action": mt5.TRADE_ACTION_MODIFY,
            "position": ticket,
            "symbol": self.symbol,
            "magic": 234000,
            "type_time": mt5.ORDER_TIME_GTC
        }
        
        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp
            
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Failed to modify position: {result.comment}")

    def start(self, symbol, lot_size=0.1):
        """Start the trading bot"""
        try:
            self.logger.info(f"Starting trading bot for {symbol}")
            
            # Initialize MT5
            if not mt5.initialize():
                error = mt5.last_error()
                self.logger.error(f"MetaTrader5 initialization failed: {error}")
                raise Exception(f"MetaTrader5 initialization failed: {error}")

            if not mt5.symbol_select(symbol, True):
                error = mt5.last_error()
                self.logger.error(f"Failed to select symbol {symbol}: {error}")
                raise Exception(f"Symbol {symbol} not found: {error}")

            # Load models and set parameters
            self._load_models(symbol)
            self.symbol = symbol
            self.lot_size = lot_size
            
            self.is_running = True
            self.thread = threading.Thread(target=self._trading_loop)
            self.thread.daemon = True
            self.thread.start()
            
            self.logger.info(f"Trading bot started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting trading bot: {str(e)}")
            raise

    def stop(self):
        """Stop the trading bot"""
        try:
            self.logger.info(f"Stopping trading bot for {self.symbol}")
            self.is_running = False
            self._close_all_positions()
            
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=30)
            
            mt5.shutdown()
            self.logger.info("Trading bot stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping trading bot: {str(e)}")
            raise
