from modules.data_processor import DataProcessor
from modules.model_trainer import ModelTrainer
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import time
import os
def initialize_mt5():
    """Initialize MetaTrader5 with explicit path and parameters"""
    if not mt5.initialize(
        path=r"C:\Program Files\MetaTrader 5\terminal64.exe",
        login=42692833,
        password="g1eqVin@7vSB5!",
        server="AdmiralsGroup-Demo",
        timeout=30000
    ):
        print(f"MetaTrader5 initialization failed. Error: {mt5.last_error()}")
        return False
    
    # Wait for terminal to fully initialize
    time.sleep(5)
    
    # Print account info to verify connection
    account_info = mt5.account_info()
    if account_info is not None:
        print(f"Connected to account #{account_info.login} {account_info.server}")
    
    # Verify symbols availability
    symbols = ["EURUSD", "BTCUSD"]
    for symbol in symbols:
        if not mt5.symbol_select(symbol, True):
            print(f"Failed to select {symbol} symbol. Error: {mt5.last_error()}")
            mt5.shutdown()
            return False
    
    return True

def train_models():
    print("Starting model training process...")
    
    # Initialize MetaTrader5
    print("Initializing MetaTrader5...")
    if not initialize_mt5():
        return False
    
    try:
        # Initialize components
        data_processor = DataProcessor()
        model_trainer = ModelTrainer()
        
        # Set date range for historical data (last 6 months)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        # Train models for each symbol
        symbols = ["EURUSD", "BTCUSD"]
        for symbol in symbols:
            print(f"\nTraining models for {symbol}...")
            
            print(f"Fetching historical data for {symbol}...")
            historical_data = data_processor.get_historical_data(symbol, start_date, end_date)
            print(f"Got {len(historical_data)} historical data points")
            
            print("Fetching news data...")
            news_data = data_processor.get_news_data(symbol)
            print(f"Got {len(news_data)} news items")
            
            # Prepare combined dataset
            print("Preparing features and labels...")
            combined_data = data_processor.prepare_features(historical_data, news_data)
            labeled_data = data_processor.create_labels(combined_data)
            
            # Create symbol-specific model directory
            model_dir = f"models/{symbol.lower()}"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            # Train models
            print(f"Training historical model for {symbol}...")
            model_trainer.train_historical_model(labeled_data, model_dir)
            
            print(f"Training news model for {symbol}...")
            model_trainer.train_news_model(news_data, model_dir)
            
            print(f"Training combined model for {symbol}...")
            model_trainer.train_combined_model(labeled_data, news_data, model_dir)
            
            print(f"Model training completed for {symbol}!")
        
        print("\nAll model training completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return False
    
    finally:
        # Always shut down MT5 connection
        mt5.shutdown()

if __name__ == "__main__":
    train_models()
