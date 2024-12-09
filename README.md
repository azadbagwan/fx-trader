# FX-Trader AI Platform

An advanced forex trading platform that combines machine learning with technical analysis and news sentiment to make trading decisions. The platform integrates with MetaTrader 5 for executing trades and provides a web interface for monitoring and control.

## Features

- Machine learning models for price prediction using historical data
- News sentiment analysis for market insight
- Real-time trading execution through MetaTrader 5
- Web-based dashboard for monitoring and control
- Combined model approach using both technical and fundamental analysis
- Automated risk management with configurable stop-loss and take-profit levels

## Prerequisites

- Python 3.8+
- MetaTrader 5 installed and configured with a trading account
- News API key for sentiment analysis

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fx-trader
```

2. Create a virtual environment and activate it:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip3 install -r requirements.txt
```

## Project Structure

```
fx-trader/
├── app.py                 # Flask application entry point
├── requirements.txt       # Project dependencies
├── README.md             # Project documentation
├── modules/
│   ├── data_processor.py # Data processing and feature engineering
│   ├── model_trainer.py  # ML model training and management
│   └── trader.py         # Trading execution and management
└── templates/
    └── index.html        # Web interface template
```

## Usage

1. Start the Flask application:
```bash
python3 app.py
```

2. Access the web interface at `http://localhost:5000`

3. Select a trading pair and configure training parameters

4. Train the models using historical data

5. Start the trading bot when ready

## Trading Strategy

The platform uses a sophisticated multi-model approach:

1. **Historical Data Model**: LSTM-based model analyzing technical indicators
   - Moving averages
   - RSI
   - MACD
   - Bollinger Bands
   - Volume indicators

2. **News Sentiment Model**: Analyzes news sentiment for trading pairs
   - Uses natural language processing
   - Considers recent news impact
   - Weighs sentiment scores

3. **Combined Decision Making**: Weighted combination of both models
   - Configurable weights for technical vs news analysis
   - Risk management rules
   - Position sizing logic

## Risk Management

- Automated stop-loss and take-profit levels
- Position sizing based on account balance
- Maximum drawdown protection
- Trading session restrictions
- Multiple timeframe analysis

## Performance Monitoring

The web interface provides real-time monitoring of:
- Total profit/loss
- Win rate
- Number of trades
- Maximum drawdown
- Performance charts
- Trading status

## Safety Features

- Automatic stop-loss for all trades
- Maximum position size limits
- Emergency stop functionality
- Error handling and logging
- Account protection measures

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Trading forex carries significant risks, and you should carefully consider whether trading is appropriate for you in light of your experience, objectives, financial resources, and other circumstances.

## Support

For support, please open an issue in the repository or contact the development team.
