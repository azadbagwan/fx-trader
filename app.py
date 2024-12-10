from flask import Flask, render_template, request, jsonify
from modules.trader import TradingBot
from modules.model_trainer import ModelTrainer
from modules.data_processor import DataProcessor
from datetime import datetime
import os
import traceback

app = Flask(__name__)

# Initialize components
data_processor = DataProcessor()
model_trainer = ModelTrainer()
trading_bots = {}  # Dictionary to store multiple bot instances

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    try:
        data = request.get_json()
        ticker = data.get('ticker')
        
        # Parse dates from frontend format (YYYY-MM-DD) to datetime objects
        try:
            start_date = datetime.strptime(data.get('start_date'), '%Y-%m-%d')
            end_date = datetime.strptime(data.get('end_date'), '%Y-%m-%d')
        except ValueError as e:
            return jsonify({
                'status': 'error',
                'message': f'Invalid date format: {str(e)}'
            }), 400
        
        # Ensure end date is not before start date
        if end_date < start_date:
            return jsonify({
                'status': 'error',
                'message': 'End date must be after start date'
            }), 400
        
        # Create model directory if it doesn't exist
        model_dir = f'models/{ticker.lower()}'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Process historical data
        print(f"Fetching historical data for {ticker}...")
        historical_data = data_processor.get_historical_data(ticker, start_date, end_date)
        print(f"Got {len(historical_data)} historical data points")
        
        print("Fetching news data...")
        news_data = data_processor.get_news_data(ticker)
        print(f"Got {len(news_data)} news items")
        
        # Prepare data
        print("Preparing features and labels...")
        combined_data = data_processor.prepare_features(historical_data, news_data)
        labeled_data = data_processor.create_labels(combined_data)
        
        # Train models
        print("Training historical model...")
        model_trainer.train_historical_model(labeled_data, model_dir)
        
        print("Training news model...")
        model_trainer.train_news_model(news_data, model_dir)
        
        print("Training combined model...")
        model_trainer.train_combined_model(labeled_data, news_data, model_dir)
        
        return jsonify({
            'status': 'success',
            'message': 'Model training completed successfully',
            'details': {
                'historical_data_points': len(historical_data),
                'news_items': len(news_data)
            }
        })
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error during training: {str(e)}\n{error_trace}")
        return jsonify({
            'status': 'error',
            'message': f'Training failed: {str(e)}',
            'trace': error_trace
        }), 500

@app.route('/start_trading', methods=['POST'])
def start_trading():
    try:
        data = request.get_json()
        ticker = data.get('ticker')
        risk_level = data.get('risk_level', 'medium')
        
        # Check if bot already exists for this ticker
        if ticker in trading_bots:
            return jsonify({
                'status': 'error',
                'message': f'Trading bot for {ticker} is already running'
            }), 400
        
        # Check if models exist for this ticker
        model_dir = f'models/{ticker.lower()}'
        required_files = [
            'historical_model.keras',
            'news_model.keras',
            'combined_model.keras',
            'historical_scaler.pkl'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
        if missing_files:
            return jsonify({
                'status': 'error',
                'message': f'Missing required model files for {ticker}: {", ".join(missing_files)}. Please train the models first.'
            }), 400
        
        # Initialize new trading bot with specified risk level
        trading_bots[ticker] = TradingBot(risk_level=risk_level)
        
        # Start the bot
        trading_bots[ticker].start(ticker)
        
        return jsonify({
            'status': 'success',
            'message': f'Trading bot started successfully for {ticker}',
            'risk_level': risk_level
        })
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error starting trading bot: {str(e)}\n{error_trace}")
        if ticker in trading_bots:
            try:
                trading_bots[ticker].stop()
            except:
                pass
            del trading_bots[ticker]
        return jsonify({
            'status': 'error',
            'message': f'Failed to start trading: {str(e)}',
            'trace': error_trace
        }), 500

@app.route('/stop_trading', methods=['POST'])
def stop_trading():
    try:
        data = request.get_json()
        ticker = data.get('ticker')
        
        if ticker not in trading_bots:
            return jsonify({
                'status': 'error',
                'message': f'No active trading bot found for {ticker}'
            }), 400
        
        trading_bots[ticker].stop()
        del trading_bots[ticker]
        return jsonify({
            'status': 'success',
            'message': f'Trading bot stopped successfully for {ticker}'
        })
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error stopping trading bot: {str(e)}\n{error_trace}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to stop trading: {str(e)}',
            'trace': error_trace
        }), 500

@app.route('/get_performance', methods=['GET'])
def get_performance():
    try:
        ticker = request.args.get('ticker')
        if ticker not in trading_bots:
            return jsonify({
                'status': 'error',
                'message': f'No active trading bot for {ticker}'
            }), 400
            
        bot = trading_bots[ticker]
        performance = bot.get_performance_metrics()
        # Add current state information
        current_state = bot.get_current_state()
        if current_state:
            performance.update({
                'current_position': current_state.get('position', 'None'),
                'decision_confidence': current_state.get('decision_confidence', 0.0),
                'last_prediction': current_state.get('prediction', None),
                'last_update': current_state.get('timestamp', None),
                'risk_level': current_state.get('risk_level', 'medium')
            })
        return jsonify(performance)
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error getting performance: {str(e)}\n{error_trace}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get performance: {str(e)}',
            'trace': error_trace
        }), 500

@app.route('/get_active_bots', methods=['GET'])
def get_active_bots():
    try:
        active_bots = []
        for ticker, bot in trading_bots.items():
            state = bot.get_current_state()
            active_bots.append({
                'ticker': ticker,
                'risk_level': state.get('risk_level', 'medium'),
                'position': state.get('position', 'None'),
                'last_update': state.get('timestamp', None)
            })
        return jsonify({
            'status': 'success',
            'active_bots': active_bots
        })
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error getting active bots: {str(e)}\n{error_trace}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get active bots: {str(e)}',
            'trace': error_trace
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
