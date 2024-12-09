import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from newsapi import NewsApiClient
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

class DataProcessor:
    def __init__(self):
        self.newsapi = NewsApiClient(api_key='YOUR_NEWS_API_KEY')  # Replace with actual key
        nltk.download('vader_lexicon')
        self.sia = SentimentIntensityAnalyzer()

    def get_historical_data(self, ticker, start_date, end_date):
        """
        Fetch historical data from MetaTrader 5
        """
        if not mt5.initialize():
            raise Exception("MetaTrader5 initialization failed")

        # Convert dates to timestamp
        start_ts = pd.Timestamp(start_date).timestamp()
        end_ts = pd.Timestamp(end_date).timestamp()

        # Fetch OHLCV data
        rates = mt5.copy_rates_range(ticker, mt5.TIMEFRAME_M15, 
                                   start_ts, end_ts)
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Add technical indicators
        df = self._add_technical_indicators(df)
        
        return df

    def get_news_data(self, ticker):
        """
        Fetch and process news data for the given ticker
        """
        # Get company name from ticker (you might need a mapping dictionary)
        company = ticker.split(':')[0] if ':' in ticker else ticker

        # Fetch news
        news = self.newsapi.get_everything(
            q=company,
            language='en',
            sort_by='publishedAt',
            page_size=100
        )

        news_data = []
        for article in news['articles']:
            sentiment = self.sia.polarity_scores(article['title'] + ' ' + article['description'])
            news_data.append({
                'date': article['publishedAt'],
                'title': article['title'],
                'description': article['description'],
                'sentiment': sentiment['compound']
            })

        return pd.DataFrame(news_data)

    def _add_technical_indicators(self, df):
        """
        Add technical indicators to the dataset
        """
        # Moving averages
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

    def prepare_features(self, historical_data, news_data):
        """
        Combine historical and news data into features for the model
        """
        # Merge news sentiment with price data
        news_data['date'] = pd.to_datetime(news_data['date'])
        historical_data = historical_data.set_index('time')
        news_data = news_data.set_index('date')
        
        # Resample news sentiment to match price data frequency
        sentiment = news_data['sentiment'].resample('15T').mean()
        
        # Combine data
        combined = pd.concat([historical_data, sentiment], axis=1)
        combined = combined.fillna(method='ffill')
        
        return combined

    def create_labels(self, df, threshold=0.001):
        """
        Create labels for supervised learning
        1 for profitable long trade
        -1 for profitable short trade
        0 for hold
        """
        future_returns = df['close'].pct_change(periods=4).shift(-4)  # 1-hour future returns
        df['label'] = 0
        df.loc[future_returns > threshold, 'label'] = 1
        df.loc[future_returns < -threshold, 'label'] = -1
        
        return df
