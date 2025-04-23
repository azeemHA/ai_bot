import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_model(file_path):
    """
    Train a model to predict buy/sell signals.
    """
    data = pd.read_csv(file_path)
    data['Signal'] = (data['Returns'] > 0).astype(int)
    X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = data['Signal']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return model

if __name__ == "__main__":
    file_path = "AAPL_processed_data.csv"
    model = train_model(file_path)
    print("Model training complete.")
