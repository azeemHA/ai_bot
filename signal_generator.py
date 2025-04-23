import pandas as pd
import pickle
import numpy as np

def calculate_support_resistance(data):
    """
    Calculate support and resistance levels based on historical highs and lows.
    """
    data['Support'] = data['Low'].rolling(window=20).min()
    data['Resistance'] = data['High'].rolling(window=20).max()
    return data

def generate_signals(model, file_path):
    """
    Generate buy/sell signals using the trained model.
    """
    data = pd.read_csv(file_path)
    X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data['Signal'] = model.predict(X)
    data['Action'] = data['Signal'].apply(lambda x: 'Buy' if x == 1 else 'Sell')
    
    # Add support and resistance levels
    data = calculate_support_resistance(data)
    
    # Add stop-loss and take-profit levels
    data['Stop_Loss'] = np.where(data['Action'] == 'Buy', data['Close'] * 0.95, data['Close'] * 1.05)
    data['Take_Profit'] = np.where(data['Action'] == 'Buy', data['Close'] * 1.10, data['Close'] * 0.90)
    
    # Save the signals to a CSV file
    data.to_csv("AAPL_signals.csv", index=False)
    print("Signals generated and saved to AAPL_signals.csv")

if __name__ == "__main__":
    file_path = "AAPL_processed_data.csv"
    with open("trained_model.pkl", "rb") as f:
        model = pickle.load(f)
    generate_signals(model, file_path)
