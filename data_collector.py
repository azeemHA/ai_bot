import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data for a given ticker.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

def calculate_profitability(data):
    """
    Calculate profitability based on historical returns.
    """
    data['Daily_Return'] = data['Close'].pct_change()
    total_return = data['Daily_Return'].sum()
    return total_return

def apply_advanced_strategy(data):
    """
    Apply advanced strategies to predict top movers.
    """
    # Momentum indicators: RSI and MACD
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    data['RSI'] = 100 - (100 / (1 + gain / loss))
    
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    
    # Combine indicators into a signal
    data['Signal'] = np.where((data['RSI'] < 30) & (data['MACD'] > 0), 1, 0)
    return data['Signal'].sum()  # Sum of signals as a proxy for potential movement

def train_ml_model(data):
    """
    Train a machine learning model to predict buy/sell signals.
    """
    data['Daily_Return'] = data['Close'].pct_change()
    data['Target'] = (data['Daily_Return'] > 0).astype(int)
    data.dropna(inplace=True)
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    X = data[features]
    y = data['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    return model

def find_profitable_tickers(tickers, start_date, end_date):
    """
    Identify the most profitable tickers from a list.
    """
    profitability = {}
    for ticker in tickers:
        try:
            data = fetch_stock_data(ticker, start_date, end_date)
            profitability[ticker] = apply_advanced_strategy(data)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    sorted_tickers = sorted(profitability.items(), key=lambda x: x[1], reverse=True)
    return sorted_tickers

if __name__ == "__main__":
    # Fetch all tickers from a predefined list or API (example: S&P 500 tickers)
    tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol'].tolist()
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    profitable_tickers = find_profitable_tickers(tickers, start_date, end_date)
    
    print("Top movers identified:")
    for ticker, score in profitable_tickers[:10]:  # Display top 10 movers
        print(f"{ticker}: {score}")
    
    # Save data for the top mover and train ML model
    top_ticker = profitable_tickers[0][0]
    data = fetch_stock_data(top_ticker, start_date, end_date)
    model = train_ml_model(data)
    data.to_csv(f"{top_ticker}_data.csv", index=False)
    print(f"Data for {top_ticker} saved to {top_ticker}_data.csv")
