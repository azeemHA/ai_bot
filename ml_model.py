import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def prepare_data(data):
    """
    Prepare data for training the ML model.
    """
    # Features: Technical indicators
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['BB_upper'] = data['SMA_20'] + (data['Close'].rolling(window=20).std() * 2)
    data['BB_lower'] = data['SMA_20'] - (data['Close'].rolling(window=20).std() * 2)
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['RSI'] = 100 - (100 / (1 + data['Close'].diff().clip(lower=0).rolling(window=14).mean() /
                                data['Close'].diff().clip(upper=0).abs().rolling(window=14).mean()))
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    # Target: 1 if price increases, 0 otherwise
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

    # Drop rows with NaN values
    data = data.dropna()

    # Features and target
    features = ['SMA_20', 'BB_upper', 'BB_lower', 'EMA_12', 'EMA_26', 'MACD', 'Signal_Line', 'RSI', 'SMA_50', 'SMA_200']
    X = data[features]
    y = data['Target']
    return X, y

def train_model(X, y):
    """
    Train a Random Forest model.
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    return model, scaler, accuracy

def predict_signals(model, scaler, data):
    """
    Predict buy/sell signals using the trained model.
    """
    features = ['SMA_20', 'BB_upper', 'BB_lower', 'EMA_12', 'EMA_26', 'MACD', 'Signal_Line', 'RSI', 'SMA_50', 'SMA_200']
    X = data[features]
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    data['ML_Prediction'] = predictions
    return data
