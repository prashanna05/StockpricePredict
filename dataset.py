import os
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
import json

def create_dataset(ticker='AAPL', period='5y', interval='1d'):
    """Download and save Yahoo Finance dataset"""
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    filename = f"data/raw/{ticker}_{period}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    
    # Download data
    print(f"📥 Downloading {ticker} data ({period})...")
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    
    # Save raw data
    data.to_csv(filename)
    
    # Process for LSTM (close prices only)
    close_prices = data['Close'].values.reshape(-1, 1)
    np.save(f'data/processed/{ticker}_close_prices.npy', close_prices)
    
    # Save metadata
    metadata = {
        'ticker': ticker,
        'period': period,
        'interval': interval,
        'created': datetime.now().isoformat(),
        'shape': close_prices.shape,
        'date_range': [str(data.index[0]), str(data.index[-1])]
    }
    
    with open(f'data/{ticker}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Dataset saved: {filename}")
    print(f"📊 Shape: {close_prices.shape}")
    return data, close_prices

if __name__ == "__main__":
    create_dataset()