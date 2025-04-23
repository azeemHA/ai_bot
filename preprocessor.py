import pandas as pd

def preprocess_data(file_path):
    """
    Preprocess stock data for ML models.
    """
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data['Returns'] = data['Close'].pct_change()
    data.dropna(inplace=True)
    return data

if __name__ == "__main__":
    file_path = "AAPL_data.csv"
    processed_data = preprocess_data(file_path)
    processed_data.to_csv("AAPL_processed_data.csv")
    print("Data preprocessing complete. Saved to AAPL_processed_data.csv")
