import matplotlib.pyplot as plt
import os
from datetime import datetime  # Import for timestamp

def generate_chart(data, ticker):
    """
    Generate a chart for price movements and signals.
    """
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(data['Close'], label='Close Price', color='blue')
        plt.plot(data['SMA_20'], label='SMA 20', color='orange')
        plt.fill_between(data.index, data['BB_upper'], data['BB_lower'], color='gray', alpha=0.3, label='Bollinger Bands')
        
        # Highlight buy and sell signals
        buy_signals = data[data['Action'] == 'Buy']
        sell_signals = data[data['Action'] == 'Sell']
        plt.scatter(buy_signals.index, buy_signals['Close'], label='Buy Signal', marker='^', color='green', alpha=1)
        plt.scatter(sell_signals.index, sell_signals['Close'], label='Sell Signal', marker='v', color='red', alpha=1)
        
        plt.title(f"{ticker} Price Movements and Signals")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid()
        
        # Save the chart to a file with a timestamp
        output_dir = os.path.join(os.getcwd(), "charts")  # Ensure absolute path
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Generate timestamp
        file_path = os.path.join(output_dir, f"{ticker}_chart_{timestamp}.png")
        plt.savefig(file_path)
        plt.close()
        return file_path
    except Exception as e:
        raise RuntimeError(f"Failed to generate chart for {ticker}: {e}")
