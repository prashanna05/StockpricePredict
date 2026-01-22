# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import talib

class StockPreprocessor:
    def __init__(self, sequence_length=60, prediction_horizon=1):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = MinMaxScaler()
        
    def add_technical_indicators(self, df):
        """Add technical indicators to dataframe"""
        df = df.copy()
        
        # Ensure we have required columns
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column")
        
        # Price-based indicators
        df['Returns'] = df['Close'].pct_change()
        
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Price targets (for supervised learning)
        df['Target_Next_Close'] = df['Close'].shift(-self.prediction_horizon)
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def prepare_features(self, df):
        """Prepare feature matrix"""
        # Select features (excluding target)
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_20', 'SMA_50', 'EMA_20', 'RSI', 'MACD',
            'Signal_Line', 'MACD_Histogram', 'BB_Middle',
            'BB_Upper', 'BB_Lower', 'BB_Width', 'Volume_SMA',
            'Volume_Ratio', 'Volatility', 'Returns'
        ]
        
        # Keep only existing columns
        existing_columns = [col for col in feature_columns if col in df.columns]
        features = df[existing_columns]
        
        return features
    
    def create_sequences(self, data, target_col='Target_Next_Close'):
        """Create sequences for LSTM"""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - self.prediction_horizon):
            # Input sequence
            seq = data.iloc[i:i + self.sequence_length].values
            
            # Target value
            if target_col in data.columns:
                target = data.iloc[i + self.sequence_length][target_col]
            else:
                # If no target column, predict next close price
                target = data.iloc[i + self.sequence_length]['Close']
            
            X.append(seq)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Second split: train and validation
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, shuffle=False
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_data(self, X_train, X_val, X_test):
        """Scale the data using MinMaxScaler"""
        # Get original shapes
        n_train, seq_len, n_features = X_train.shape
        
        # Reshape for scaling (flatten sequence dimension)
        X_train_flat = X_train.reshape(-1, n_features)
        X_val_flat = X_val.reshape(-1, n_features)
        X_test_flat = X_test.reshape(-1, n_features)
        
        # Fit scaler on training data
        self.scaler.fit(X_train_flat)
        
        # Transform all datasets
        X_train_scaled = self.scaler.transform(X_train_flat).reshape(n_train, seq_len, n_features)
        X_val_scaled = self.scaler.transform(X_val_flat).reshape(X_val.shape[0], seq_len, n_features)
        X_test_scaled = self.scaler.transform(X_test_flat).reshape(X_test.shape[0], seq_len, n_features)
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def inverse_transform_target(self, y_scaled):
        """Inverse transform target values to original scale"""
        # Create dummy array with target at Close position
        dummy = np.zeros((len(y_scaled), self.scaler.n_features_in_))
        # Assuming 'Close' is at index 3 (adjust based on your feature order)
        close_idx = 3
        dummy[:, close_idx] = y_scaled
        
        # Inverse transform
        y_original = self.scaler.inverse_transform(dummy)[:, close_idx]
        
        return y_original

if __name__ == "__main__":
    # Test preprocessing
    print("Testing preprocessing module...")