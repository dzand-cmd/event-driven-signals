# src/event_driven_signals.py

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fetch_data(ticker, period="2y"):
    """
    Fetch historical price data from Yahoo Finance.
    """
    data = yf.Ticker(ticker).history(period=period)
    if data.empty:
        raise ValueError("No data fetched. Check the ticker or network connection.")
    return data

def generate_events(data):
    """
    Generate events when price crosses above 200-day SMA.
    """
    data['SMA200'] = data['Close'].rolling(200).mean()
    # Event: price crosses above SMA200
    data['Event'] = (data['Close'] > data['SMA200']) & (data['Close'].shift(1) <= data['SMA200'].shift(1))
    data['Event'] = data['Event'].fillna(False)
    return data

def process_events(data):
    """
    Convert events into signals: 1 = Buy, 0 = Hold/Sell
    """
    data['Signal'] = 0
    data.loc[data['Event'], 'Signal'] = 1
    return data

def backtest_signal(data):
    """
    Simple backtest: buy 1 unit on each signal, hold until next signal.
    Computes cumulative returns.
    """
    data['Returns'] = data['Close'].pct_change().fillna(0)
    # Strategy: apply returns only when holding position
    data['StrategyReturns'] = data['Signal'].shift(1) * data['Returns']
    data['CumulativeStrategy'] = (1 + data['StrategyReturns']).cumprod()
    data['CumulativeMarket'] = (1 + data['Returns']).cumprod()
    return data

def plot_results(data, ticker):
    plt.figure(figsize=(12,6))
    plt.plot(data.index, data['CumulativeMarket'], label='Market')
    plt.plot(data.index, data['CumulativeStrategy'], label='Strategy')
    plt.title(f'Cumulative Returns: {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    ticker = 'NVDA'
    print(f"Fetching data for {ticker}...")
    data = fetch_data(ticker)
    
    data = generate_events(data)
    data = process_events(data)
    data = backtest_signal(data)
    
    print(data[['Close', 'SMA200', 'Event', 'Signal', 'CumulativeStrategy']].tail(10))
    plot_results(data, ticker)

if __name__ == "__main__":
    main()