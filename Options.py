import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as si
import requests
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint

#Alpha Vantage API Key
API_KEY = "7WFT0JC4HSB0MMPL"  

#Fetch historical exchange rates from Alpha Vantage
def get_forex_data(from_currency, to_currency):
    url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={from_currency}&to_symbol={to_currency}&apikey={API_KEY}&outputsize=full"
    response = requests.get(url).json()
    
    if "Time Series FX (Daily)" in response:
        data = response["Time Series FX (Daily)"]
        df = pd.DataFrame(data).T  #Transpose to get date as index
        df = df[["4. close"]].astype(float)  #Keep only closing price
        df.columns = [f"{from_currency}/{to_currency}"]
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    else:
        print(f"Error fetching data for {from_currency}/{to_currency}, trying Yahoo Finance")
        return get_forex_data_yf(from_currency, to_currency)

#Fetch historical exchange rates from Yahoo Finance
def get_forex_data_yf(from_currency, to_currency):
    pair = f"{from_currency}{to_currency}=X"
    try:
        df = yf.download(pair, period="10y", interval="1d")
        df = df[["Close"]]
        df.columns = [f"{from_currency}/{to_currency}"]
        return df
    except Exception as e:
        print(f"Yahoo Finance data fetch failed for {pair}: {e}")
        return pd.DataFrame()

#Define currency pairs to fetch
target_currencies = {"INR": "USD/INR", "CNY": "USD/CNY", "RUB": "USD/RUB", "JPY": "USD/JPY"}

exchange_rates = {}
for currency, label in target_currencies.items():
    exchange_rates[label] = get_forex_data("USD", currency)

exchange_rates_df = pd.concat(exchange_rates.values(), axis=1)
#exchange_rates_df.dropna(inplace=True)
#exchange_rates_df.fillna(method='ffill', inplace=True)

#Black-Scholes-Merton Model with FX considerations
def black_scholes_merton_fx(S, K, T, r_d, r_f, sigma, option_type="call"):
    if sigma <= 0 or T <= 0:
        return 0
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * np.exp(-r_f * T) * si.norm.cdf(d1) - K * np.exp(-r_d * T) * si.norm.cdf(d2)
    else:
        price = K * np.exp(-r_d * T) * si.norm.cdf(-d2) - S * np.exp(-r_f * T) * si.norm.cdf(-d1)
    
    return price

#Binomial Tree Model
def binomial_tree_fx(S, K, T, r_d, r_f, sigma, N=100, option_type="call"):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r_d - r_f) * dt) - d) / (u - d)
    
    stock_prices = np.zeros((N+1, N+1))
    option_values = np.zeros((N+1, N+1))
    
    for i in range(N+1):
        for j in range(i+1):
            stock_prices[j, i] = S * (u ** (i-j)) * (d ** j)
    
    if option_type == "call":
        option_values[:, N] = np.maximum(stock_prices[:, N] - K, 0)
    else:
        option_values[:, N] = np.maximum(K - stock_prices[:, N], 0)
    
    for i in range(N-1, -1, -1):
        for j in range(i+1):
            option_values[j, i] = np.exp(-r_d * dt) * (p * option_values[j, i+1] + (1-p) * option_values[j+1, i+1])
    
    return option_values[0, 0]

#Statistical Arbitrage - Cointegration Test
def cointegration_test(series1, series2, significance=0.1):  #Changed from 0.05
    score, p_value, _ = coint(series1, series2)
    return p_value < significance

#Delta Hedging Strategy
def black_scholes_delta(S, K, T, r_d, r_f, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return si.norm.cdf(d1)
    else:
        return si.norm.cdf(d1) - 1

delta_hedging = exchange_rates_df.copy()
delta_hedging['Delta'] = delta_hedging.iloc[:, 0].apply(lambda x: black_scholes_delta(x, 100, 1, 0.05, 0.02, 0.2, "call"))

#Visualization of exchange rates
plt.figure(figsize=(14, 7))
for pair in exchange_rates_df.columns:
    plt.plot(exchange_rates_df.index, exchange_rates_df[pair], label=pair)
plt.title('Historical Exchange Rates')
plt.xlabel('Year')
plt.ylabel('Exchange Rate')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(delta_hedging.index, delta_hedging['Delta'], label='Delta Hedging Position')
plt.title('Delta Hedging Strategy Over Time')
plt.xlabel('Time')
plt.ylabel('Delta')
plt.legend()
plt.grid()
plt.show()
