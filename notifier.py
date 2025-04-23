import pandas as pd

def send_notifications(file_path):
    """
    Send notifications for buy/sell signals.
    """
    data = pd.read_csv(file_path)
    for index, row in data.iterrows():
        print(f"Date: {row['Date']}, Action: {row['Action']}, Close Price: {row['Close']}")

if __name__ == "__main__":
    file_path = "AAPL_signals.csv"
    send_notifications(file_path)
