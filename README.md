# AI Trading Bot

The AI Trading Bot is a Python-based system that analyzes stock market data, generates buy/sell signals, and sends notifications via Telegram. It uses technical indicators, sentiment analysis, and a machine learning model to provide actionable insights for traders.

---

## Features
- **Technical Indicators**: Bollinger Bands, MACD, RSI, and Moving Average Crossovers.
- **Sentiment Analysis**: Analyzes news headlines using VADER Sentiment Analysis.
- **Machine Learning**: Predicts buy/sell signals using a Random Forest Classifier.
- **Risk Metrics**: Calculates Sharpe Ratio and Maximum Drawdown.
- **Telegram Notifications**: Sends detailed buy/sell signals and charts to a Telegram chat.
- **Scheduled Execution**: Runs automatically on GitHub Actions every 2 hours from Monday to Friday between 8:30 AM and 3:30 PM CST.

---

## Installation

### Prerequisites
- Python 3.9 or higher
- Git
- A Telegram bot token and chat ID
- A NewsAPI key for sentiment analysis

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ai-trading-bot.git
   cd ai-trading-bot
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the project root and add the following:
     ```
     TELEGRAM_BOT_TOKEN=your-telegram-bot-token
     TELEGRAM_CHAT_ID=your-telegram-chat-id
     NEWSAPI_KEY=your-newsapi-key
     ```

5. Run the script:
   ```bash
   python main.py
   ```

---

## Usage

### Signal Messages
The bot sends buy/sell signals to your Telegram chat in the following format:

#### Example Message:
```
🤖 AI Trading Bot Signals 🤖

📊 Model Accuracy: 85.00%

📈 Buy Signals 📈

🔹 Ticker: AAPL
  💵 Current Price: $150.00
  ✅ Buy Price: $150.00
  🎯 Target Profit: $165.00
  📉 Stop Loss: $142.50
  📊 Sentiment: 0.75
  💰 P/E Ratio: 25.00
  🏢 Market Cap: 2.5T
  📈 Sharpe Ratio: 1.20
  📉 Max Drawdown: -10.00%
  💸 Dividend Yield: 0.60%
  📊 Debt-to-Equity: 1.50

📉 Sell Signals 📉

🔹 Ticker: TSLA
  💵 Current Price: $700.00
  ❌ Sell Price: $700.00
  📊 Sentiment: -0.25
  💰 P/E Ratio: 50.00
  🏢 Market Cap: 700B
  📈 Sharpe Ratio: 0.90
  📉 Max Drawdown: -15.00%
  💸 Dividend Yield: N/A
  📊 Debt-to-Equity: 2.00
```

### How to Interpret the Signals
1. **Buy Signals**:
   - **Buy Price**: The recommended price to buy the stock.
   - **Target Profit**: The price at which you should consider selling to achieve a 10% profit.
   - **Stop Loss**: The price at which you should sell to minimize losses (5% below the buy price).

2. **Sell Signals**:
   - **Sell Price**: The recommended price to sell the stock.

3. **Additional Metrics**:
   - **Sentiment**: Positive values indicate optimism; negative values indicate pessimism.
   - **P/E Ratio**: A lower value may indicate undervaluation.
   - **Market Cap**: The total market value of the company.
   - **Sharpe Ratio**: Higher values indicate better risk-adjusted returns.
   - **Max Drawdown**: The maximum observed loss from a peak.
   - **Dividend Yield**: The annual dividend payout as a percentage of the stock price.
   - **Debt-to-Equity**: A measure of the company's financial leverage.

---

## Scheduling with GitHub Actions

The bot is configured to run automatically using GitHub Actions. It executes every 2 hours from Monday to Friday between 8:30 AM and 3:30 PM CST.

### Workflow File
The workflow file is located at `.github/workflows/schedule.yml` and uses the following schedule:
```yaml
on:
  schedule:
    - cron: '30 14-21/2 * * 1-5'
```

---

## Contributing
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature-name"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.