import logging
from data_collector import find_profitable_tickers, train_ml_model
import pandas as pd
from datetime import datetime, timedelta
import requests
import os
import yfinance as yf  # Add this import for fetching current prices
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # Import VADER
import numpy as np  # Add this import for risk metrics calculations
from visualization import generate_chart  # Import for chart generation
from ml_model import prepare_data, train_model, predict_signals  # Import ML functions

# Setup logger
def setup_logger():
    """
    Setup logger for the AI bot.
    """
    logging.basicConfig(
        filename="ai_bot.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger()

logger = setup_logger()

def send_telegram_notification(message, bot_token, chat_id):
    """
    Send a notification message via Telegram.
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        logger.info("Notification sent successfully.")
    else:
        logger.error(f"Failed to send notification: {response.text}")

def send_telegram_chart(file_path, bot_token, chat_id):
    """
    Send a chart image via Telegram.
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    with open(file_path, 'rb') as photo:
        response = requests.post(url, data={"chat_id": chat_id}, files={"photo": photo})
    if response.status_code == 200:
        logger.info("Chart sent successfully.")
    else:
        logger.error(f"Failed to send chart: {response.text}")

def fetch_stock_data_safe(ticker, start_date, end_date):
    """
    Fetch historical stock data using yfinance.
    """
    try:
        logger.info(f"Fetching data for ticker: {ticker}")
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)  # Removed unsupported 'progress' argument
        if data.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        return data
    except Exception as e:
        logger.error(f"Failed to fetch historical data for ticker {ticker}: {e}")
        return None

def validate_telegram_credentials(bot_token, chat_id):
    """
    Validate Telegram bot token and chat ID.
    """
    if not bot_token or not chat_id:
        logger.error("Telegram bot token or chat ID is missing. Please set them as environment variables.")
        raise ValueError("Telegram bot token or chat ID is missing.")

def analyze_ticker(data):
    """
    Analyze a ticker's data and generate buy/sell signals using Bollinger Bands, MACD, RSI, and Moving Average Crossovers.
    """
    # Ensure required indicators are calculated
    if 'SMA_20' not in data.columns:
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
    if 'BB_upper' not in data.columns or 'BB_lower' not in data.columns:
        data['BB_upper'] = data['SMA_20'] + (data['Close'].rolling(window=20).std() * 2)
        data['BB_lower'] = data['SMA_20'] - (data['Close'].rolling(window=20).std() * 2)
    if 'EMA_12' not in data.columns or 'EMA_26' not in data.columns:
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    if 'EMA_26' not in data.columns:
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    if 'MACD' not in data.columns or 'Signal_Line' not in data.columns:
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    if 'RSI' not in data.columns:
        data['RSI'] = 100 - (100 / (1 + data['Close'].diff().clip(lower=0).rolling(window=14).mean() /
                                    data['Close'].diff().clip(upper=0).abs().rolling(window=14).mean()))
    if 'SMA_50' not in data.columns:
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
    if 'SMA_200' not in data.columns:
        data['SMA_200'] = data['Close'].rolling(window=200).mean()

    # Generate buy/sell signals
    data['Action'] = None  # Ensure the 'Action' column exists
    buy_signals = data[(data['RSI'] < 30) & (data['Close'] < data['BB_lower']) & (data['MACD'] > data['Signal_Line'])]
    sell_signals = data[(data['RSI'] > 70) & (data['Close'] > data['BB_upper']) & (data['MACD'] < data['Signal_Line'])]
    data.loc[data.index.isin(buy_signals.index), 'Action'] = 'Buy'
    data.loc[data.index.isin(sell_signals.index), 'Action'] = 'Sell'
    
    logger.info(f"Generated {len(buy_signals)} buy signals and {len(sell_signals)} sell signals.")
    return buy_signals, sell_signals

def calculate_risk_metrics(data):
    """
    Calculate risk management metrics like Sharpe Ratio and Maximum Drawdown.
    """
    returns = data['Close'].pct_change()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized Sharpe Ratio
    cumulative_returns = (1 + returns).cumprod()
    max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
    return sharpe_ratio, max_drawdown

def fetch_fundamental_metrics(ticker):
    """
    Fetch additional fundamental metrics like Dividend Yield and Debt-to-Equity Ratio.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        dividend_yield = info.get('dividendYield', 'N/A')
        debt_to_equity = info.get('debtToEquity', 'N/A')
        return dividend_yield, debt_to_equity
    except Exception as e:
        logger.error(f"Failed to fetch fundamental metrics for {ticker}: {e}")
        return 'N/A', 'N/A'

# Initialize VADER Sentiment Analyzer
vader_analyzer = SentimentIntensityAnalyzer()

def perform_sentiment_analysis(ticker):
    """
    Perform sentiment analysis for the given ticker using VADER and real news headlines.
    """
    try:
        # Fetch news headlines using NewsAPI
        api_key = os.getenv("NEWSAPI_KEY")  # Fetch NewsAPI key from environment variable
        url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}"
        response = requests.get(url)
        articles = response.json().get('articles', [])
        
        # Extract headlines
        headlines = [article['title'] for article in articles[:5]]  # Limit to 5 headlines
        
        # Analyze sentiment for each headline
        sentiment_scores = [vader_analyzer.polarity_scores(headline)['compound'] for headline in headlines]
        
        # Calculate average sentiment score
        average_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
        return average_sentiment
    except Exception as e:
        logger.error(f"Failed to perform sentiment analysis for {ticker}: {e}")
        return 0.0  # Neutral sentiment in case of error

def fetch_financial_data(ticker):
    """
    Fetch financial data for the given ticker.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        pe_ratio = info.get('forwardPE', 'N/A')
        market_cap = info.get('marketCap', 'N/A')
        return pe_ratio, market_cap
    except Exception as e:
        logger.error(f"Failed to fetch financial data for {ticker}: {e}")
        return 'N/A', 'N/A'

def fetch_current_price(ticker):
    """
    Fetch the current price of a ticker from the market.
    """
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.history(period="1d")['Close'].iloc[-1]
        return current_price
    except Exception as e:
        logger.error(f"Failed to fetch current price for {ticker}: {e}")
        return None

def fetch_all_tickers():
    """
    Fetch tickers from multiple US markets.
    """
    try:
        # Fetch S&P 500 tickers
        sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol'].tolist()
        
        # Fetch NASDAQ-100 tickers
        nasdaq_tables = pd.read_html("https://en.wikipedia.org/wiki/NASDAQ-100")
        for table in nasdaq_tables:
            if 'Ticker' in table.columns or 'Company' in table.columns:
                nasdaq = table['Ticker'].tolist() if 'Ticker' in table.columns else table['Company'].tolist()
                break
        else:
            raise ValueError("Could not find NASDAQ-100 tickers in the webpage.")
        
        # Combine and deduplicate tickers
        return list(set(sp500 + nasdaq))
    except Exception as e:
        logger.error(f"Failed to fetch tickers: {e}")
        return []

def portfolio_allocation(tickers):
    """
    Implement a basic portfolio allocation strategy.
    """
    allocation = {}
    total_weight = 1.0  # 100% of the portfolio
    weight_per_ticker = total_weight / len(tickers)
    for ticker in tickers:
        allocation[ticker] = weight_per_ticker
    return allocation

def main():
    """
    Main function to run the AI bot workflow.
    """
    logger.info("Starting AI bot workflow.")
    
    # Fetch sensitive credentials from environment variables
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    newsapi_key = os.getenv("NEWSAPI_KEY")

    # Validate credentials
    validate_telegram_credentials(bot_token, chat_id)
    if not newsapi_key:
        logger.error("NewsAPI key is missing. Please set it as an environment variable.")
        raise ValueError("NewsAPI key is missing.")

    # Step 1: Fetch all tickers (e.g., S&P 500 tickers and NASDAQ tickers)
    logger.info("Fetching tickers...")
    tickers = fetch_all_tickers()
    logger.info(f"Fetched {len(tickers)} tickers.")
    
    # Step 2: Define date range for analysis (last 3 years)
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=3*365)).strftime('%Y-%m-%d')
    logger.info(f"Using date range: {start_date} to {end_date}")
    
    # Step 3: Identify the most profitable tickers
    logger.info("Identifying top movers...")
    profitable_tickers = find_profitable_tickers(tickers, start_date, end_date)
    logger.info(f"Identified top 10 movers: {[ticker for ticker, _ in profitable_tickers[:10]]}")
    
    # Step 4: Analyze each top ticker and generate buy/sell signals
    
    # Portfolio allocation
    allocation = portfolio_allocation([ticker for ticker, _ in profitable_tickers[:10]])
    logger.info(f"Portfolio allocation: {allocation}")

    # Train ML model
    logger.info("Training ML model...")
    sample_ticker = "AAPL"  # Use a sample ticker for training
    sample_data = fetch_stock_data_safe(sample_ticker, start_date, end_date)
    if sample_data is not None:
        X, y = prepare_data(sample_data)
        ml_model, scaler, model_accuracy = train_model(X, y)  # Get model accuracy
    else:
        logger.error("Failed to fetch data for ML model training.")
        return

    consolidated_message = f"🤖 *AI Trading Bot Signals* 🤖\n\n"
    consolidated_message += f"📊 *Model Accuracy*: {model_accuracy:.2%}\n\n"  # Add model accuracy
    buy_message = "📈 *Buy Signals* 📈\n\n"
    sell_message = "📉 *Sell Signals* 📉\n\n"
    signals_found = False

    for ticker, _ in profitable_tickers[:10]:
        logger.info(f"Processing ticker: {ticker}")
        data = fetch_stock_data_safe(ticker, start_date, end_date)
        if data is None:
            logger.error(f"Skipping ticker {ticker} due to data fetch failure.")
            continue

        try:
            buy_signals, sell_signals = analyze_ticker(data)
            current_price = fetch_current_price(ticker)
            sentiment_score = perform_sentiment_analysis(ticker)
            pe_ratio, market_cap = fetch_financial_data(ticker)
            dividend_yield, debt_to_equity = fetch_fundamental_metrics(ticker)
            sharpe_ratio, max_drawdown = calculate_risk_metrics(data)

            # Use current price for sell signals in the graph and message
            if not sell_signals.empty:
                sell_signals['Close'] = current_price

            # Send chart only if signals are generated
            if not buy_signals.empty or not sell_signals.empty:
                chart_path = generate_chart(data, ticker)
                send_telegram_chart(chart_path, bot_token, chat_id)

            # Target profit and stop loss
            if current_price is not None:
                buy_price = current_price
                stop_loss = buy_price * 0.95  # 5% below the buy price
                target_profit = buy_price * 1.10  # 10% above the buy price

                if not buy_signals.empty:
                    buy_message += f"🔹 *Ticker*: {ticker}\n"
                    buy_message += f"  💵 *Current Price*: ${current_price:.2f}\n"
                    buy_message += f"  ✅ *Buy Price*: ${buy_price:.2f}\n"
                    buy_message += f"  🎯 *Target Profit*: ${target_profit:.2f}\n"
                    buy_message += f"  📉 *Stop Loss*: ${stop_loss:.2f}\n"
                    buy_message += f"  📊 *Sentiment*: {sentiment_score:.2f}\n"
                    buy_message += f"  💰 *P/E Ratio*: {pe_ratio}\n"
                    buy_message += f"  🏢 *Market Cap*: {market_cap}\n"
                    buy_message += f"  📈 *Sharpe Ratio*: {sharpe_ratio:.2f}\n"
                    buy_message += f"  📉 *Max Drawdown*: {max_drawdown:.2%}\n"
                    buy_message += f"  💸 *Dividend Yield*: {dividend_yield}\n"
                    buy_message += f"  📊 *Debt-to-Equity*: {debt_to_equity}\n\n"
                    signals_found = True

                if not sell_signals.empty:
                    sell_price = current_price  # Use current price for sell signal
                    sell_message += f"🔹 *Ticker*: {ticker}\n"
                    sell_message += f"  💵 *Current Price*: ${current_price:.2f}\n"
                    sell_message += f"  ❌ *Sell Price*: ${sell_price:.2f}\n"
                    sell_message += f"  📊 *Sentiment*: {sentiment_score:.2f}\n"
                    sell_message += f"  💰 *P/E Ratio*: {pe_ratio}\n"
                    sell_message += f"  🏢 *Market Cap*: {market_cap}\n"
                    sell_message += f"  📈 *Sharpe Ratio*: {sharpe_ratio:.2f}\n"
                    sell_message += f"  📉 *Max Drawdown*: {max_drawdown:.2%}\n"
                    sell_message += f"  💸 *Dividend Yield*: {dividend_yield}\n"
                    sell_message += f"  📊 *Debt-to-Equity*: {debt_to_equity}\n\n"
                    signals_found = True
        except Exception as e:
            logger.error(f"Error analyzing ticker {ticker}: {e}")
            continue
    
    # Combine buy and sell messages
    final_message = consolidated_message + buy_message + sell_message
    if signals_found:
        logger.info("Sending consolidated notification.")
        try:
            send_telegram_notification(final_message, bot_token, chat_id)
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    else:
        logger.info("No buy/sell signals generated. No notification sent.")

if __name__ == "__main__":
    main()
    logger.info("Workflow complete.")
