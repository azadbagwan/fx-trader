# FX-Trader AI Platform

An advanced forex trading platform that combines machine learning with technical analysis and news sentiment to make trading decisions. The platform integrates with MetaTrader 5 for executing trades and provides a web interface for monitoring and control.

## How It Works

1. **Model Training**
   - Historical price data analysis
   - Technical indicators (SMA, RSI, MACD, etc.)
   - News sentiment analysis
   - Combined decision making

2. **Trading Process**
   - Analyzes market every 60 seconds
   - Makes predictions (Short/Hold/Long)
   - Shows confidence level for each decision
   - Executes trades when confidence > 60%

3. **Trading Rules**
   - Short Position: Opens when Short probability > 60%
   - Long Position: Opens when Long probability > 60%
   - Hold: No trading when confidence ≤ 60% or Hold is predicted
   - All trades include stop-loss and take-profit orders

4. **Web Interface Elements**
   - Current Position: Shows if bot is Short/Hold/Long
   - Confidence Level: Shows prediction confidence
   - Probabilities: Shows % for each position type
   - Performance Metrics: Profit/Loss, Win Rate, etc.

## Setup Requirements

1. MetaTrader 5 Terminal
   - Must be installed and running
   - Need valid trading account (demo or real)
   - Terminal must be logged in

2. Python Environment
   - Python 3.8+ with required packages
   - MetaTrader5 Python package
   - TensorFlow for predictions

## Using the Platform

1. **Start Trading**
   - Click "Start Trading" in web interface
   - Bot connects to MetaTrader
   - Begins analyzing market
   - Shows predictions and confidence

2. **Monitor Trades**
   - Web Interface: Shows current position, confidence, metrics
   - MetaTrader: Shows actual trades, profit/loss
   - Trading log: Shows detailed decision process

3. **Stop Trading**
   - Click "Stop Trading" to halt
   - Closes all open positions
   - Saves performance metrics

## Understanding the Display

1. **Position Indicator**
   - Red: Short position
   - Yellow: Hold (no position)
   - Green: Long position

2. **Confidence Bar**
   - Shows prediction confidence
   - Trades only execute above 60%
   - Higher confidence = stronger signal

3. **Performance Metrics**
   - Total Profit/Loss
   - Number of trades
   - Win rate percentage
   - Current positions

## Trading Log

The bot maintains detailed logs showing:
- Current market price
- Prediction probabilities
- Trading decisions
- Order execution details
- Position updates

Check `trading_bot.log` for detailed operation history.

## Safety Features

- Stop-loss on all trades
- Take-profit targets
- Confidence thresholds
- Position size limits
- Error handling
- Automatic position tracking

## Important Notes

1. Always monitor the bot's operation
2. Check MetaTrader to verify trades
3. Use demo account for testing
4. Understand the risks involved
5. Monitor the log file for details

## Troubleshooting

If trades aren't appearing:
1. Check MetaTrader is running
2. Verify account login
3. Check trading permissions
4. Monitor confidence levels
5. Check error logs
