import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from newsapi import NewsApiClient
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
import joblib
import os

class DataProcessor:
    def __init__(self):
        self.newsapi = NewsApiClient(api_key='c12bbd6f608c47f6b0c1c3c3c83f961f')
        nltk.download('vader_lexicon', quiet=True)
        self.sia = SentimentIntensityAnalyzer()
        self.scaler = StandardScaler()
        self.recent_patterns = []  # Store recent price patterns
        self.market_state = {
            'regime': 'normal',
            'volatility': 0.0,
            'trend_strength': 0.0
        }

    def get_historical_data(self, ticker, start_date, end_date):
        """Fetch and process historical data with enhanced features"""
        if not mt5.initialize():
            raise Exception("MetaTrader5 initialization failed")

        # Convert dates to timestamp
        start_ts = pd.Timestamp(start_date).timestamp()
        end_ts = pd.Timestamp(end_date).timestamp()

        # Fetch data at 5-minute intervals for more granular analysis
        rates = mt5.copy_rates_range(ticker, mt5.TIMEFRAME_M5, 
                                   start_ts, end_ts)
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        
        # Add enhanced technical indicators
        df = self._add_technical_indicators(df, ticker)
        
        # Add market regime features
        df = self._add_regime_features(df)
        
        return df

    def _add_technical_indicators(self, df, ticker):
        """Enhanced technical indicators with focus on short-term movements"""
        # Short-term momentum
        df['Returns'] = df['close'].pct_change()
        df['Momentum_1h'] = df['close'].pct_change(12)  # 12 5-min bars = 1 hour
        df['Momentum_15m'] = df['close'].pct_change(3)  # 3 5-min bars = 15 min
        
        # Volatility measures
        df['ATR'] = self._calculate_atr(df)
        df['Volatility'] = df['Returns'].rolling(window=12).std()
        
        # Volume analysis
        df['Volume_MA5'] = df['tick_volume'].rolling(window=5).mean()
        df['Volume_MA15'] = df['tick_volume'].rolling(window=15).mean()
        df['Volume_Ratio'] = df['tick_volume'] / df['Volume_MA15']
        
        # Price channels
        df['Upper_Channel'] = df['high'].rolling(window=20).max()
        df['Lower_Channel'] = df['low'].rolling(window=20).min()
        df['Channel_Position'] = (df['close'] - df['Lower_Channel']) / (df['Upper_Channel'] - df['Lower_Channel'])
        
        # Momentum indicators
        df['RSI'] = self._calculate_rsi(df['close'])
        df['RSI_MA'] = df['RSI'].rolling(window=5).mean()
        df['RSI_Divergence'] = df['RSI'] - df['RSI_MA']
        
        # Trend strength
        df['ADX'] = self._calculate_adx(df)
        
        # Custom indicators for crypto
        if ticker == "BTCUSD":
            # Add crypto-specific indicators
            df['Price_Range'] = (df['high'] - df['low']) / df['low'] * 100
            df['Volume_Price_Trend'] = df['tick_volume'] * df['Returns']
            df['Buying_Pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        return df

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI with smoothing"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=period).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.ewm(span=period).mean()

    def _calculate_adx(self, df, period=14):
        """Calculate Average Directional Index"""
        # Calculate +DM and -DM
        high_diff = df['high'].diff()
        low_diff = df['low'].diff()
        
        pos_dm = high_diff.where((high_diff > 0) & (high_diff > low_diff), 0)
        neg_dm = -low_diff.where((low_diff > 0) & (low_diff > high_diff), 0)
        
        # Calculate TR
        tr = self._calculate_atr(df, period)
        
        # Calculate +DI and -DI
        pos_di = 100 * (pos_dm.ewm(span=period).mean() / tr)
        neg_di = 100 * (neg_dm.ewm(span=period).mean() / tr)
        
        # Calculate DX and ADX
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.ewm(span=period).mean()
        
        return adx

    def _add_regime_features(self, df):
        """Add market regime indicators"""
        # Volatility regime
        df['Volatility_Regime'] = pd.qcut(df['Volatility'].rolling(window=60).mean(), 
                                        q=3, labels=['low', 'medium', 'high'])
        
        # Trend regime
        df['Trend_Strength'] = df['ADX'].rolling(window=12).mean()
        df['Trend_Regime'] = pd.qcut(df['Trend_Strength'], 
                                   q=3, labels=['ranging', 'weak_trend', 'strong_trend'])
        
        # Volume regime
        df['Volume_Regime'] = pd.qcut(df['Volume_Ratio'].rolling(window=12).mean(),
                                    q=3, labels=['low', 'normal', 'high'])
        
        return df

    def get_news_data(self, ticker):
        """Enhanced news data processing with real-time sentiment adjustment"""
        search_terms = {
            "BTCUSD": ['Bitcoin', 'BTC', 'cryptocurrency', 'crypto market', 'blockchain'],
            "EURUSD": ['EUR/USD', 'euro dollar', 'ECB', 'Federal Reserve', 'forex']
        }.get(ticker, [ticker])

        all_news = []
        for term in search_terms:
            news = self.newsapi.get_everything(
                q=term,
                language='en',
                sort_by='publishedAt',
                page_size=100
            )
            
            for article in news['articles']:
                # Enhanced sentiment analysis
                title_sentiment = self.sia.polarity_scores(article['title'])
                desc_sentiment = self.sia.polarity_scores(article['description'] or '')
                
                # Weight recent news more heavily
                time_weight = self._calculate_time_weight(article['publishedAt'])
                
                # Combine sentiments with time weight
                combined_sentiment = (
                    title_sentiment['compound'] * 0.7 + 
                    desc_sentiment['compound'] * 0.3
                ) * time_weight
                
                # Add source credibility
                source = article['source']['name'].lower()
                credibility = self._get_source_credibility(source, ticker)
                
                # Final weighted sentiment
                weighted_sentiment = combined_sentiment * credibility
                
                all_news.append({
                    'date': pd.to_datetime(article['publishedAt']).tz_convert('UTC'),
                    'title': article['title'],
                    'sentiment': weighted_sentiment,
                    'source': source,
                    'credibility': credibility,
                    'time_weight': time_weight
                })

        news_df = pd.DataFrame(all_news)
        if not news_df.empty:
            news_df.set_index('date', inplace=True)
            # Resample to 5-minute intervals for more frequent updates
            aggregated = news_df.resample('5T').agg({
                'sentiment': 'mean',
                'credibility': 'mean',
                'time_weight': 'mean'
            }).fillna(method='ffill')
            
            return aggregated
        return pd.DataFrame()

    def _calculate_time_weight(self, published_at):
        """Calculate time-based weight for news articles"""
        published_time = pd.to_datetime(published_at).tz_convert('UTC')
        age_hours = (datetime.now(published_time.tzinfo) - published_time).total_seconds() / 3600
        
        # Exponential decay with 6-hour half-life
        return np.exp(-age_hours / 6)

    def _get_source_credibility(self, source, ticker):
        """Enhanced source credibility scoring"""
        base_credibility = {
            "BTCUSD": {
                'coindesk': 1.0,
                'cointelegraph': 0.9,
                'bitcoin magazine': 0.9,
                'decrypt': 0.8,
                'the block': 0.9,
                'bloomberg': 0.85,
                'reuters': 0.85,
                'forbes': 0.8
            },
            "EURUSD": {
                'reuters': 1.0,
                'bloomberg': 1.0,
                'financial times': 0.95,
                'wall street journal': 0.95,
                'cnbc': 0.85,
                'marketwatch': 0.85,
                'forex live': 0.8,
                'dailyfx': 0.8
            }
        }
        
        return base_credibility.get(ticker, {}).get(source, 0.6)

    def create_labels(self, df, ticker):
        """Dynamic label creation based on market conditions"""
        # Calculate forward returns at different horizons
        for horizon in [1, 3, 6, 12]:  # 5min, 15min, 30min, 1hour
            df[f'fwd_return_{horizon}'] = df['close'].pct_change(horizon).shift(-horizon)
        
        # Dynamic threshold based on volatility
        volatility = df['Returns'].rolling(window=12).std()
        base_threshold = volatility.rolling(window=60).mean()
        
        if ticker == "BTCUSD":
            # More aggressive thresholds for BTC
            threshold = base_threshold * 0.5  # 0.5x the rolling volatility
        else:
            threshold = base_threshold * 0.8  # 0.8x the rolling volatility
        
        # Create labels for each horizon
        labels = []
        for horizon in [1, 3, 6, 12]:
            fwd_return = df[f'fwd_return_{horizon}']
            horizon_labels = pd.Series(1, index=df.index)  # Default to hold
            
            # Dynamic thresholds based on horizon
            horizon_threshold = threshold * np.sqrt(horizon)
            
            # Create labels
            horizon_labels[fwd_return > horizon_threshold] = 2  # Long
            horizon_labels[fwd_return < -horizon_threshold] = 0  # Short
            
            labels.append(horizon_labels)
        
        # Combine labels (majority vote)
        df['label'] = pd.concat(labels, axis=1).mode(axis=1)[0]
        
        return df

    def prepare_features(self, historical_data, news_data):
        """Prepare features with enhanced feature engineering"""
        # Ensure both dataframes have UTC timezone
        historical_data = historical_data.set_index('time')
        
        # Combine data
        combined = pd.concat([historical_data, news_data], axis=1)
        
        # Fill missing values
        combined = combined.fillna(method='ffill')
        
        # Add interaction features
        if 'sentiment' in combined.columns:
            combined['Sentiment_Momentum'] = combined['sentiment'] * combined['Momentum_1h']
            combined['Sentiment_Volume'] = combined['sentiment'] * combined['Volume_Ratio']
        
        # Add time-based features
        combined['Hour'] = combined.index.hour
        combined['DayOfWeek'] = combined.index.dayofweek
        
        return combined

    def update_market_state(self, current_data):
        """Update market state based on recent data"""
        self.market_state['volatility'] = current_data['Volatility'].iloc[-1]
        self.market_state['trend_strength'] = current_data['ADX'].iloc[-1]
        
        # Determine market regime
        if self.market_state['volatility'] > current_data['Volatility'].quantile(0.8):
            self.market_state['regime'] = 'volatile'
        elif self.market_state['trend_strength'] > 25:
            self.market_state['regime'] = 'trending'
        else:
            self.market_state['regime'] = 'normal'
        
        return self.market_state

    def save_scaler(self, symbol):
        """Save the fitted scaler"""
        if not os.path.exists(f'models/{symbol.lower()}'):
            os.makedirs(f'models/{symbol.lower()}')
        joblib.dump(self.scaler, f'models/{symbol.lower()}/scaler.pkl')
