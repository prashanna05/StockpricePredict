# src/data_loader.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class YahooFinanceDataLoader:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        
    def download_data(self):
        """Download stock data from Yahoo Finance"""
        print(f"Downloading data for {self.ticker}...")
        self.data = yf.download(
            self.ticker, 
            start=self.start_date, 
            end=self.end_date,
            progress=False
        )
        print(f"Data downloaded: {len(self.data)} records")
        return self.data
    
    def save_to_csv(self, filepath):
        """Save data to CSV file"""
        if self.data is not None:
            self.data.to_csv(filepath)
            print(f"Data saved to {filepath}")
        else:
            print("No data to save. Download data first.")
    
    def load_from_csv(self, filepath):
        """Load data from CSV file"""
        self.data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"Data loaded from {filepath}: {len(self.data)} records")
        return self.data
    
    def get_metadata(self):
        """Get metadata about the stock"""
        if self.data is not None:
            return {
                'ticker': self.ticker,
                'start_date': self.data.index[0],
                'end_date': self.data.index[-1],
                'num_records': len(self.data),
                'columns': list(self.data.columns)
            }
        return None

if __name__ == "__main__":
    # Test the data loader
    loader = YahooFinanceDataLoader("AAPL", "2020-01-01", "2023-12-31")
    data = loader.download_data()
    print(data.head())