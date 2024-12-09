from flask import Flask, render_template, request, jsonify
from modules.trader import TradingBot
from modules.model_trainer import ModelTrainer
from modules.data_processor import DataProcessor
import MetaTrader5 as mt5
import os

app = Flask(__name__)

# Initialize components
data_processor = DataProcessor()
model_trainer = ModelTrainer()
trading_bot = TradingBot()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    data = request.get_json()
    ticker = data.get('ticker')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    
    # Process historical data
    historical_data = data_processor.get_historical_data(ticker, start_date, end_date)
    news_data = data_processor.get_news_data(ticker)
    
    # Train models
    model_trainer.train_historical_model(historical_data)
    model_trainer.train_news_model(news_data)
    
    return jsonify({'status': 'success', 'message': 'Model training completed'})

@app.route('/start_trading', methods=['POST'])
def start_trading():
    data = request.get_json()
    ticker = data.get('ticker')
    
    if not mt5.initialize():
        return jsonify({'status': 'error', 'message': 'MetaTrader5 initialization failed'})
    
    trading_bot.start(ticker)
    return jsonify({'status': 'success', 'message': 'Trading bot started'})

@app.route('/stop_trading', methods=['POST'])
def stop_trading():
    trading_bot.stop()
    return jsonify({'status': 'success', 'message': 'Trading bot stopped'})

@app.route('/get_performance', methods=['GET'])
def get_performance():
    performance = trading_bot.get_performance_metrics()
    return jsonify(performance)

if __name__ == '__main__':
    app.run(debug=True)
