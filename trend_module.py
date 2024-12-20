
from datetime import date,datetime,timedelta
import requests
from requests.adapters import HTTPAdapter, Retry
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas_ta as ta
import json
import os

# Get the absolute path to the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
secrets_path = os.path.join(current_dir, 'secrets.txt')

# Read and parse the file
with open(secrets_path, 'r') as f:
    secrets = json.load(f)  # Safely evaluate the contents as a dictionary

# Get the API key
ALPHA_VANTAGE_API_KEY = secrets['ALPHA_VANTAGE_API_KEY']
MAX_CALLS_PER_MINUTE = 70



def calculate_percentiles(series, window=100, suffix=""):
    """
    Calculate rolling 30th and 70th percentiles over a specified window.
    """

    percentiles = pd.DataFrame(index=series.index)
    percentiles[f'{suffix}50th_Percentile'] = series.rolling(window, min_periods=1).quantile(0.5)
    percentiles[f'{suffix}70th_Percentile'] = series.rolling(window, min_periods=1).quantile(0.7)
    return percentiles


def calculate_atr(df, window=100):
    """
    Calculate the Average True Range (ATR) over a specified window.
    """
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    # True Range
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    # Average True Range
    atr = tr.rolling(window).mean()
    return atr


def calculate_aroon(df, window=25):
    """
    Calculate the Aroon Up and Down indicators.
    """
    aroon_up = 100 * df['High'].rolling(window + 1).apply(lambda x: x.argmax()) / window
    aroon_down = 100 * df['Low'].rolling(window + 1).apply(lambda x: x.argmin()) / window
    return aroon_up, aroon_down




def hull_moving_average(series, window):
    half_length = window // 2
    sqrt_length = int(window**0.5)
    wma1 = series.rolling(window=half_length).mean()  # First WMA
    wma2 = series.rolling(window=window).mean()       # Second WMA
    hull = 2 * wma1 - wma2                            # Hull transformation
    hma = hull.rolling(window=sqrt_length).mean()     # Final HMA
    return hma


def requests_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504), session=None):
    session = session or requests.Session()
    retry = Retry(total=retries, read=retries, connect=retries, backoff_factor=backoff_factor, status_forcelist=status_forcelist)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def get_alpha_vantage_data(symbol):
    """
    Fetch daily stock data for a given symbol from Alpha Vantage.
    """
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
    try:
        response = requests_retry_session().get(url, timeout=10, verify=False)
        response.raise_for_status()
        data = response.json()
        if "Time Series (Daily)" in data:
            return data["Time Series (Daily)"]
        else:
            print(f"Error in response: {data.get('Error Message', 'No time series data available.')}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    return None

def alpha_vantage_to_dataframe(data):
    """
    Convert Alpha Vantage time series data to a pandas DataFrame.
    """
    records = []
    for date_str, daily_data in data.items():
        records.append({
            'Date': pd.to_datetime(date_str),
            'Open': float(daily_data['1. open']),
            'High': float(daily_data['2. high']),
            'Low': float(daily_data['3. low']),
            'Close': float(daily_data['4. close']),
            'Volume': int(daily_data['5. volume'])
        })
    
    df = pd.DataFrame(records)
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

def trend_analysis(ticker='SPY'):
    start = date.today() - timedelta(days=800)
    end = date.today()

    # Fetch data from Alpha Vantage
    raw_data = get_alpha_vantage_data(ticker)
    if raw_data is None:
        print(f"Failed to retrieve data for {ticker}")
        return None
    
    df = alpha_vantage_to_dataframe(raw_data)
    
    # Filter the DataFrame to the specified date range
    #df = df[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
    #df.index = pd.to_datetime(df.index)

    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['SMA100'] = df['Close'].rolling(window=100).mean()
    df['Hull50'] = hull_moving_average(df['Close'], 50)
    df['Hull50_Trend'] = df['Hull50'].pct_change()
    df['Trendline'] = df['SMA100']
    df['ROC_Trend'] = df['Trendline'].pct_change().rolling(window=3).mean()
    df['DonchHigh20'] = df['High'].rolling(window=20).max()
    df['DonchHigh50'] = df['High'].rolling(window=50).max()
    df['DonchLow50'] = df['Low'].rolling(window=50).min()
    df['DonchLow20'] = df['Low'].rolling(window=20).min()
    df['DonchHigh200'] = df['Close'].rolling(window=200).max()
    df['DonchLow200'] = df['Close'].rolling(window=200).min()
    df['Expansion'] = df['DonchHigh200'].rolling(window=20).mean() - df['DonchLow200'].rolling(window=20).mean()
    df['Sideways'] = df['Expansion'].rolling(window=100).mean()
    psar = ta.psar(high=df['High'], low=df['Low'], af0=0.02, af=0.02, max_af=0.2)
    df = pd.concat([df, psar], axis=1)
    df['Hist_Volatility'] = df['Close'].rolling(20).std() * (252**0.8)
    df['ATR_100'] = calculate_atr(df, 100)
    df['Aroon_Up'], df['Aroon_Down'] = calculate_aroon(df, window=25)
    df['ROC_ATR'] = df['ATR_100'].pct_change()
    # Calculate percentiles for the ATR if needed
    iv_percentiles = calculate_percentiles(df['Hist_Volatility'], 100, suffix="hist_")
    # Create a summary table for the most recent values
    summary = {
        'Ticker': ticker,
        'SMA200': df['SMA200'].iloc[-1],
        'SMA100': df['SMA100'].iloc[-1],
        'Hull50': df['Hull50'].iloc[-1],
        'Hull50_Trend': df['Hull50_Trend'].iloc[-1],
        'Trendline': df['Trendline'].iloc[-1],
        'ROC_Trend': df['ROC_Trend'].iloc[-1],
        'DonchHigh20': df['DonchHigh20'].iloc[-1],
        'DonchHigh50': df['DonchHigh50'].iloc[-1],
        'DonchLow50': df['DonchLow50'].iloc[-1],
        'DonchLow20': df['DonchLow20'].iloc[-1],
        'DonchHigh200': df['DonchHigh200'].iloc[-1],
        'DonchLow200': df['DonchLow200'].iloc[-1],
        'Expansion': df['Expansion'].iloc[-1],
        'Sideways': df['Sideways'].iloc[-1],
        'Hist_Volatility': df['Hist_Volatility'].iloc[-1],
        'ATR_100': df['ATR_100'].iloc[-1],
        'Aroon_Up': df['Aroon_Up'].iloc[-1],
        'Aroon_Down': df['Aroon_Down'].iloc[-1],
        'ROC_ATR': df['ROC_ATR'].iloc[-1],
        'Put_Spread_Open': (df['Aroon_Up'].iloc[-1] > 50 and df['Close'].iloc[-1] > df['SMA200'].iloc[-1] and df['Close'].iloc[-1] < df['Hull50'].iloc[-1] and df['ROC_Trend'].iloc[-1] > 0),
        'Call_Spread_Open': (df['Aroon_Down'].iloc[-1] > 50 and df['Close'].iloc[-1] < df['SMA200'].iloc[-1] and df['Close'].iloc[-1] > df['Hull50'].iloc[-1] and df['ROC_Trend'].iloc[-1] < 0),
    }
    return summary
# List of tickers to run the backtest on

# # List of tickers to run the backtest on 
 

# tickers = ['SPY'

#     #  

#     #

# ]


tickers = ['SPY', 'FEZ', 'FXE', 'FXI', 'GLD','BITO', 'EDC', 'EFA', 'EWJ', 'EWU', 'EWW', 'EWY', 'EWZ','KWEB', 'LIT', 'MJ', 'SLV', 'SMH', 'TMF', 'TQQQ', 'TUR'
           , 'UNG','URA', 'USO', 'UUP', 'XLB', 'XLE', 'XLF', 'XLU', 'XLV', 'XLY', 'XME', 'XOP']
# Run the backtest for each ticker and collect summaries
summaries = []

for ticker in tickers:
    summary = trend_analysis(ticker)
    summaries.append(summary)

# Convert summaries to DataFrame
summary_df = pd.DataFrame(summaries)

# Plot the closing prices for the past 100 days for each ETF
#for ticker in tickers:
#    df = yf.download(ticker, start="2024-08-22", end="2024-11-30")

#     plt.figure(figsize=(10, 5))

#     plt.plot(df.index, df['Close'], label=f'{ticker} Closing Prices')

#     plt.title(f'{ticker} Closing Prices (Last 100 Days)')

#     plt.xlabel('Date')

#     plt.ylabel('Close Price')

#     plt.legend()

#     plt.show()


# Save the summary DataFrame to a CSV file
summary_df.to_csv("etf_summary.csv", index=False)
print(summary_df)

