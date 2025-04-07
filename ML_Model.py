import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import coint, adfuller
import seaborn as sns
from Options import black_scholes_merton_fx, binomial_tree_fx, get_forex_data, get_forex_data_yf
import random

#Load multiple Forex Data for pair trading
print("Loading forex data...")
forex_pairs = {
    "USD/INR": get_forex_data("USD", "INR"),
    "USD/CNY": get_forex_data("USD", "CNY"),
    "USD/JPY": get_forex_data("USD", "JPY"),
    "USD/RUB": get_forex_data("USD", "RUB")
}

#Combine all data into a single dataframe
combined_data = pd.concat([df for df in forex_pairs.values()], axis=1)
combined_data.dropna(inplace=True)
print(f"Loaded data with shape: {combined_data.shape}")

#Calculate returns for all pairs
for pair in combined_data.columns:
    combined_data[f'{pair}_Returns'] = combined_data[pair].pct_change()

combined_data.dropna(inplace=True)

#1. Statistical Arbitrage Implementation
def find_cointegrated_pairs(dataframe):
    n = dataframe.shape[1]
    pvalue_matrix = np.ones((n, n))
    keys = dataframe.columns
    pairs = []
    
    for i in range(n):
        for j in range(i+1, n):
            #Only compare original price columns (not derived columns like returns)
            if "_" not in keys[i] and "_" not in keys[j]:
                result = coint(dataframe[keys[i]], dataframe[keys[j]])
                pvalue_matrix[i, j] = result[1]
                if result[1] < 0.05:
                    pairs.append((keys[i], keys[j], result[1]))
    
    return pvalue_matrix, pairs

#Test for cointegration
price_columns = [col for col in combined_data.columns if "_" not in col]
price_data = combined_data[price_columns]
pvalue_matrix, pairs = find_cointegrated_pairs(price_data)

#Visualize the cointegration matrix
plt.figure(figsize=(10, 8))
sns.heatmap(pvalue_matrix, xticklabels=price_columns, yticklabels=price_columns, 
            cmap='RdYlGn_r', mask=(pvalue_matrix >= 0.05))
plt.title('Cointegration p-values between currency pairs')
plt.tight_layout()
plt.savefig('cointegration_heatmap.png')
print(f"Found {len(pairs)} cointegrated pairs: {pairs}")

#Implement pair trading strategy (if cointegrated pairs exist)
if pairs:
    #Use the first cointegrated pair for demonstration
    pair1, pair2, pvalue = pairs[0]
    
    #Calculate the spread and z-score
    combined_data['Spread'] = combined_data[pair1] / combined_data[pair2]
    combined_data['Spread_Mean'] = combined_data['Spread'].rolling(window=30).mean()
    combined_data['Spread_Std'] = combined_data['Spread'].rolling(window=30).std()
    combined_data['Z_Score'] = (combined_data['Spread'] - combined_data['Spread_Mean']) / combined_data['Spread_Std']
    
    #Trading signals: buy spread when z-score < -2, sell when z-score > 2
    combined_data['Stat_Arb_Signal'] = 0
    combined_data.loc[combined_data['Z_Score'] < -2, 'Stat_Arb_Signal'] = 1    # Long spread
    combined_data.loc[combined_data['Z_Score'] > 2, 'Stat_Arb_Signal'] = -1    # Short spread
    
    #Visualize the spread and trading signals
    plt.figure(figsize=(14, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(combined_data.index, combined_data[pair1], label=pair1)
    plt.plot(combined_data.index, combined_data[pair2], label=pair2)
    plt.title(f'Price Series: {pair1} vs {pair2}')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(combined_data.index, combined_data['Spread'], label='Spread')
    plt.plot(combined_data.index, combined_data['Spread_Mean'], label='Mean (30-day)')
    plt.plot(combined_data.index, combined_data['Spread_Mean'] + 2*combined_data['Spread_Std'], 'r--', label='+2σ')
    plt.plot(combined_data.index, combined_data['Spread_Mean'] - 2*combined_data['Spread_Std'], 'r--', label='-2σ')
    plt.title('Normalized Spread')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(combined_data.index, combined_data['Z_Score'], label='Z-Score')
    plt.axhline(y=0, linestyle='-', color='k')
    plt.axhline(y=2, linestyle='--', color='r')
    plt.axhline(y=-2, linestyle='--', color='r')
    plt.fill_between(combined_data.index, 
                     combined_data['Z_Score'].values, 
                     2, 
                     where=(combined_data['Z_Score'] > 2),
                     color='red', alpha=0.3)
    plt.fill_between(combined_data.index, 
                     combined_data['Z_Score'].values, 
                     -2, 
                     where=(combined_data['Z_Score'] < -2),
                     color='green', alpha=0.3)
    plt.title('Z-Score with Trading Signals')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('statistical_arbitrage.png')
    print("Created statistical arbitrage visualization")


#2. Momentum Trading with Enhanced Indicators
#Choose USD/INR for individual analysis
forex_data = forex_pairs["USD/INR"].copy()
forex_data['Returns'] = forex_data.pct_change()

#Moving Averages
forex_data['50_MA'] = forex_data['USD/INR'].rolling(window=50).mean()
forex_data['200_MA'] = forex_data['USD/INR'].rolling(window=200).mean()

#RSI Calculation
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

forex_data['RSI'] = calculate_rsi(forex_data['USD/INR'])

#MACD Calculation
forex_data['12_EMA'] = forex_data['USD/INR'].ewm(span=12, adjust=False).mean()
forex_data['26_EMA'] = forex_data['USD/INR'].ewm(span=26, adjust=False).mean()
forex_data['MACD'] = forex_data['12_EMA'] - forex_data['26_EMA']
forex_data['Signal_Line'] = forex_data['MACD'].ewm(span=9, adjust=False).mean()
forex_data['MACD_Histogram'] = forex_data['MACD'] - forex_data['Signal_Line']

#Combined Momentum Signals
forex_data.dropna(inplace=True)
forex_data['MA_Signal'] = np.where(forex_data['50_MA'] > forex_data['200_MA'], 1, -1)
forex_data['RSI_Signal'] = np.where(forex_data['RSI'] < 30, 1, np.where(forex_data['RSI'] > 70, -1, 0))
forex_data['MACD_Signal'] = np.where(forex_data['MACD'] > forex_data['Signal_Line'], 1, -1)

#Combined signal (ensemble)
forex_data['Momentum_Signal'] = (forex_data['MA_Signal'] + forex_data['RSI_Signal'] + forex_data['MACD_Signal']) / 3
forex_data['Position'] = forex_data['Momentum_Signal'].apply(lambda x: 1 if x > 0.3 else (-1 if x < -0.3 else 0))

#Visualize momentum signals
plt.figure(figsize=(16, 12))

plt.subplot(4, 1, 1)
plt.plot(forex_data.index, forex_data['USD/INR'], label='USD/INR')
plt.plot(forex_data.index, forex_data['50_MA'], label='50-day MA')
plt.plot(forex_data.index, forex_data['200_MA'], label='200-day MA')
plt.fill_between(forex_data.index, 
                 forex_data['USD/INR'].values, 
                 forex_data['50_MA'].values, 
                 where=(forex_data['MA_Signal'] > 0),
                 color='green', alpha=0.3)
plt.fill_between(forex_data.index, 
                 forex_data['USD/INR'].values, 
                 forex_data['50_MA'].values, 
                 where=(forex_data['MA_Signal'] < 0),
                 color='red', alpha=0.3)
plt.title('USD/INR with Moving Averages')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(forex_data.index, forex_data['RSI'], label='RSI')
plt.axhline(y=70, linestyle='--', color='r')
plt.axhline(y=30, linestyle='--', color='g')
plt.fill_between(forex_data.index, forex_data['RSI'], 70, 
                 where=(forex_data['RSI'] > 70), 
                 color='red', alpha=0.3)
plt.fill_between(forex_data.index, forex_data['RSI'], 30, 
                 where=(forex_data['RSI'] < 30), 
                 color='green', alpha=0.3)
plt.title('Relative Strength Index (RSI)')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(forex_data.index, forex_data['MACD'], label='MACD')
plt.plot(forex_data.index, forex_data['Signal_Line'], label='Signal Line')
plt.bar(forex_data.index, forex_data['MACD_Histogram'], color=np.where(forex_data['MACD_Histogram'] > 0, 'g', 'r'), width=3)
plt.title('MACD Indicator')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(forex_data.index, forex_data['Momentum_Signal'], label='Combined Signal')
plt.axhline(y=0.3, linestyle='--', color='g')
plt.axhline(y=-0.3, linestyle='--', color='r')
plt.fill_between(forex_data.index, forex_data['Momentum_Signal'], 0.3, 
                 where=(forex_data['Momentum_Signal'] > 0.3),
                 color='green', alpha=0.3)
plt.fill_between(forex_data.index, forex_data['Momentum_Signal'], -0.3, 
                 where=(forex_data['Momentum_Signal'] < -0.3),
                 color='red', alpha=0.3)
plt.title('Combined Momentum Signal')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('momentum_trading.png')
print("Created momentum trading visualization")

#3. Order Book Analysis (Enhanced Simulation)
class OrderBook:
    def __init__(self, initial_price, spread_factor=0.0001, depth=5, volatility=0.0005):
        self.mid_price = initial_price
        self.spread_factor = spread_factor
        self.depth = depth
        self.volatility = volatility
        self.orders = {
            'bid': [],
            'ask': []
        }
        self.update_orders()
    
    def update_orders(self):
        spread = self.mid_price * self.spread_factor
        self.bid_price = self.mid_price - spread/2
        self.ask_price = self.mid_price + spread/2
        
        #Generate bid and ask book with decreasing sizes
        self.orders['bid'] = [(self.bid_price - i*spread/self.depth, 
                              np.random.poisson(100/(i+1))) for i in range(self.depth)]
        self.orders['ask'] = [(self.ask_price + i*spread/self.depth, 
                              np.random.poisson(100/(i+1))) for i in range(self.depth)]
    
    def update_price(self, price_change=None):
        if price_change is None:
            #Random walk
            price_change = np.random.normal(0, self.volatility * self.mid_price)
        self.mid_price += price_change
        self.update_orders()
    
    def execute_market_order(self, side, size):
        executed_price = 0
        remaining_size = size
        execution_cost = 0
        
        if side == 'buy':
            for price, available_size in self.orders['ask']:
                if remaining_size <= 0:
                    break
                executed = min(remaining_size, available_size)
                execution_cost += executed * price
                remaining_size -= executed
            #Market impact - push prices up slightly
            self.update_price(self.volatility * self.mid_price * size / 100)
        else:  #sell
            for price, available_size in self.orders['bid']:
                if remaining_size <= 0:
                    break
                executed = min(remaining_size, available_size)
                execution_cost += executed * price
                remaining_size -= executed
            #Market impact - push prices down slightly
            self.update_price(-self.volatility * self.mid_price * size / 100)
        
        #Return average execution price
        if size - remaining_size > 0:
            avg_price = execution_cost / (size - remaining_size)
            return avg_price, size - remaining_size
        return self.mid_price, 0

#Simulate order book for a day of trading
def simulate_order_book(initial_price, num_steps=100):
    book = OrderBook(initial_price)
    mid_prices = [book.mid_price]
    bid_prices = [book.bid_price]
    ask_prices = [book.ask_price]
    spreads = [book.ask_price - book.bid_price]
    market_depths = []
    
    #Calculate initial market depth (total liquidity)
    bid_depth = sum(size for _, size in book.orders['bid'])
    ask_depth = sum(size for _, size in book.orders['ask'])
    market_depths.append(bid_depth + ask_depth)
    
    #Simulate random orders and price movements
    for _ in range(num_steps):
        #Random market order
        if np.random.random() < 0.3:  #30% chance of market order
            side = 'buy' if np.random.random() < 0.5 else 'sell'
            size = np.random.randint(1, 50)
            book.execute_market_order(side, size)
        else:
            #Just update price (simulating other market factors)
            book.update_price()
        
        mid_prices.append(book.mid_price)
        bid_prices.append(book.bid_price)
        ask_prices.append(book.ask_price)
        spreads.append(book.ask_price - book.bid_price)
        
        #Calculate market depth
        bid_depth = sum(size for _, size in book.orders['bid'])
        ask_depth = sum(size for _, size in book.orders['ask'])
        market_depths.append(bid_depth + ask_depth)
    
    return {
        'mid_prices': mid_prices,
        'bid_prices': bid_prices,
        'ask_prices': ask_prices,
        'spreads': spreads,
        'market_depths': market_depths
    }

#Run order book simulation
print("Simulating order book...")
order_book_sim = simulate_order_book(forex_data['USD/INR'].iloc[-1], num_steps=200)

#Visualize order book simulation
plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(order_book_sim['mid_prices'], label='Mid Price')
plt.plot(order_book_sim['bid_prices'], label='Bid Price', linestyle='--')
plt.plot(order_book_sim['ask_prices'], label='Ask Price', linestyle='--')
plt.fill_between(range(len(order_book_sim['bid_prices'])), 
                order_book_sim['bid_prices'], 
                order_book_sim['ask_prices'], 
                color='gray', alpha=0.2)
plt.title('Simulated Order Book Prices')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(order_book_sim['spreads'], color='purple')
plt.title('Bid-Ask Spread')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(order_book_sim['market_depths'], color='green')
plt.title('Market Depth (Total Available Liquidity)')
plt.grid(True)

plt.tight_layout()
plt.savefig('order_book_simulation.png')
print("Created order book visualization")

#4. Full Greeks Calculation with 3D Visualization
def compute_all_greeks(S, K, T, r_d, r_f, sigma, option_type="call"):
    #Small increment for numerical derivatives
    dS = 0.01 * S
    dsigma = 0.001
    dT = 1/365  #One day
    
    #Base price
    price = black_scholes_merton_fx(S, K, T, r_d, r_f, sigma, option_type)
    
    #Delta: First derivative w.r.t underlying price
    delta = (black_scholes_merton_fx(S + dS, K, T, r_d, r_f, sigma, option_type) - 
             black_scholes_merton_fx(S - dS, K, T, r_d, r_f, sigma, option_type)) / (2 * dS)
    
    #Gamma: Second derivative w.r.t underlying price
    gamma = (black_scholes_merton_fx(S + dS, K, T, r_d, r_f, sigma, option_type) - 
            2 * price + 
            black_scholes_merton_fx(S - dS, K, T, r_d, r_f, sigma, option_type)) / (dS ** 2)
    
    #Vega: First derivative w.r.t volatility
    vega = (black_scholes_merton_fx(S, K, T, r_d, r_f, sigma + dsigma, option_type) - 
           black_scholes_merton_fx(S, K, T, r_d, r_f, sigma - dsigma, option_type)) / (2 * dsigma) / 100  #Divided by 100 for 1% change
    
    #Theta: First derivative w.r.t time to maturity
    theta = (black_scholes_merton_fx(S, K, T - dT, r_d, r_f, sigma, option_type) - price) / dT / 365  #Daily theta
    
    #Rho: First derivative w.r.t interest rate
    dr = 0.0001
    rho_domestic = (black_scholes_merton_fx(S, K, T, r_d + dr, r_f, sigma, option_type) - 
                   black_scholes_merton_fx(S, K, T, r_d - dr, r_f, sigma, option_type)) / (2 * dr) / 100  #Divided by 100 for 1% change
    
    rho_foreign = (black_scholes_merton_fx(S, K, T, r_d, r_f + dr, sigma, option_type) - 
                  black_scholes_merton_fx(S, K, T, r_d, r_f - dr, sigma, option_type)) / (2 * dr) / 100  #Divided by 100 for 1% change
    
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho_domestic': rho_domestic,
        'rho_foreign': rho_foreign
    }
#####
#Calculate Greeks for various prices and volatilities
print("Calculating option Greeks...")
prices = np.linspace(80, 120, 20)
volatilities = np.linspace(0.1, 0.4, 20)
time_to_maturity = 1.0  # 1 year

#Create meshgrid for 3D plots
price_grid, vol_grid = np.meshgrid(prices, volatilities)
call_price_grid = np.zeros_like(price_grid)
delta_grid = np.zeros_like(price_grid)
gamma_grid = np.zeros_like(price_grid)
vega_grid = np.zeros_like(price_grid)
theta_grid = np.zeros_like(price_grid)

for i in range(len(prices)):
    for j in range(len(volatilities)):
        greeks = compute_all_greeks(prices[i], 100, time_to_maturity, 0.05, 0.02, volatilities[j], "call")
        call_price_grid[j, i] = greeks['price']
        delta_grid[j, i] = greeks['delta']
        gamma_grid[j, i] = greeks['gamma']
        vega_grid[j, i] = greeks['vega']
        theta_grid[j, i] = greeks['theta']

#3D Visualization of option price and Greeks
fig = plt.figure(figsize=(18, 12))

#Option Price Surface
ax1 = fig.add_subplot(231, projection='3d')
surf1 = ax1.plot_surface(price_grid, vol_grid, call_price_grid, cmap='viridis')
ax1.set_xlabel('Underlying Price')
ax1.set_ylabel('Volatility')
ax1.set_zlabel('Option Price')
ax1.set_title('Call Option Price Surface')
fig.colorbar(surf1, ax=ax1, shrink=0.5)

#Delta Surface
ax2 = fig.add_subplot(232, projection='3d')
surf2 = ax2.plot_surface(price_grid, vol_grid, delta_grid, cmap='plasma')
ax2.set_xlabel('Underlying Price')
ax2.set_ylabel('Volatility')
ax2.set_zlabel('Delta')
ax2.set_title('Delta Surface')
fig.colorbar(surf2, ax=ax2, shrink=0.5)

#Gamma Surface
ax3 = fig.add_subplot(233, projection='3d')
surf3 = ax3.plot_surface(price_grid, vol_grid, gamma_grid, cmap='inferno')
ax3.set_xlabel('Underlying Price')
ax3.set_ylabel('Volatility')
ax3.set_zlabel('Gamma')
ax3.set_title('Gamma Surface')
fig.colorbar(surf3, ax=ax3, shrink=0.5)

#Vega Surface
ax4 = fig.add_subplot(234, projection='3d')
surf4 = ax4.plot_surface(price_grid, vol_grid, vega_grid, cmap='magma')
ax4.set_xlabel('Underlying Price')
ax4.set_ylabel('Volatility')
ax4.set_zlabel('Vega')
ax4.set_title('Vega Surface')
fig.colorbar(surf4, ax=ax4, shrink=0.5)

#Theta Surface
ax5 = fig.add_subplot(235, projection='3d')
surf5 = ax5.plot_surface(price_grid, vol_grid, theta_grid, cmap='cividis')
ax5.set_xlabel('Underlying Price')
ax5.set_ylabel('Volatility')
ax5.set_zlabel('Theta')
ax5.set_title('Theta Surface')
fig.colorbar(surf5, ax=ax5, shrink=0.5)

plt.tight_layout()
plt.savefig('option_greeks_3d.png')
print("Created 3D Greeks visualization")

#5. Monte Carlo Simulations for Hedging Strategies
def monte_carlo_hedging_simulation(S0, K, T, r_d, r_f, sigma, num_paths=1000, num_steps=252):
    dt = T / num_steps
    paths = np.zeros((num_paths, num_steps + 1))
    paths[:, 0] = S0
    
    #Generate random paths
    for i in range(num_paths):
        for j in range(num_steps):
            z = np.random.standard_normal()
            paths[i, j+1] = paths[i, j] * np.exp((r_d - r_f - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    
    #Calculate option price at each step for each path
    option_prices = np.zeros_like(paths)
    delta_values = np.zeros_like(paths)
    
    for i in range(num_paths):
        for j in range(num_steps + 1):
            remaining_time = T - j * dt
            if remaining_time < 0.01:  #Near expiry
                if paths[i, j] > K:  #In the money
                    option_prices[i, j] = paths[i, j] - K
                    delta_values[i, j] = 1
                else:  #Out of the money
                    option_prices[i, j] = 0
                    delta_values[i, j] = 0
            else:
                greeks = compute_all_greeks(paths[i, j], K, remaining_time, r_d, r_f, sigma, "call")
                option_prices[i, j] = greeks['price']
                delta_values[i, j] = greeks['delta']
    
    #Calculate PnL for different hedging strategies
    no_hedge_pnl = option_prices[:, -1] - option_prices[:, 0]
    
    #Delta hedging at different frequencies
    daily_hedge_pnl = np.zeros(num_paths)
    weekly_hedge_pnl = np.zeros(num_paths)
    monthly_hedge_pnl = np.zeros(num_paths)
    
    for i in range(num_paths):
        #Daily hedging (every step)
        delta_position = 0
        pnl = -option_prices[i, 0]  #Initial short position
        
        for j in range(num_steps):
            #PnL from delta position
            if j > 0:
                delta_pnl = delta_position * (paths[i, j] - paths[i, j-1])
                pnl += delta_pnl
            
            #Rebalance delta
            new_delta_position = delta_values[i, j]
            delta_adjustment = new_delta_position - delta_position
            pnl -= delta_adjustment * paths[i, j]  #Cost of rebalancing
            delta_position = new_delta_position
        
        #Final settlement
        pnl += option_prices[i, -1]  #Option expires
        pnl -= delta_position * paths[i, -1]  #Close delta position
        daily_hedge_pnl[i] = pnl
        
        #Weekly hedging (every 5 steps)
        delta_position = 0
        pnl = -option_prices[i, 0]  #Initial short position
        
        for j in range(0, num_steps, 5):
            #PnL from delta position
            if j > 0:
                delta_pnl = delta_position * (paths[i, j] - paths[i, j-5])
                pnl += delta_pnl
            
            #Rebalance delta
            new_delta_position = delta_values[i, j]
            delta_adjustment = new_delta_position - delta_position
            pnl -= delta_adjustment * paths[i, j]  #Cost of rebalancing
            delta_position = new_delta_position
        
        #Final settlement
        pnl += option_prices[i, -1]  #Option expires
        pnl -= delta_position * paths[i, -1]  #Close delta position
        weekly_hedge_pnl[i] = pnl
        
        #Monthly hedging (every 21 steps)
        delta_position = 0
        pnl = -option_prices[i, 0]  #Initial short position
        
        for j in range(0, num_steps, 21):
            #PnL from delta position
            if j > 0:
                delta_pnl = delta_position * (paths[i, j] - paths[i, max(0, j-21)])
                pnl += delta_pnl
            
            #Rebalance delta
            new_delta_position = delta_values[i, j]
            delta_adjustment = new_delta_position - delta_position
            pnl -= delta_adjustment * paths[i, j]  #Cost of rebalancing
            delta_position = new_delta_position
        
        #Final settlement
        pnl += option_prices[i, -1]  #Option expires
        pnl -= delta_position * paths[i, -1]  #Close delta position
        monthly_hedge_pnl[i] = pnl
    
    return {
        'paths': paths,
        'option_prices': option_prices,
        'delta_values': delta_values,
        'no_hedge_pnl': no_hedge_pnl,
        'daily_hedge_pnl': daily_hedge_pnl,
        'weekly_hedge_pnl': weekly_hedge_pnl,
        'monthly_hedge_pnl': monthly_hedge_pnl
    }

#Run Monte Carlo simulation
print("Running Monte Carlo hedging simulation...")
mc_results = monte_carlo_hedging_simulation(100, 100, 1, 0.05, 0.02, 0.2, num_paths=500, num_steps=252)

#Visualize Monte Carlo simulation
#Visualize Monte Carlo simulation results
plt.figure(figsize=(15, 12))

#Sample of price paths
plt.subplot(2, 2, 1)
for i in range(20):  #Plot a subset of paths
    plt.plot(mc_results['paths'][i], alpha=0.5, linewidth=0.8)
plt.title('Sample FX Price Paths')
plt.xlabel('Trading Days')
plt.ylabel('Exchange Rate')
plt.grid(True)

#PnL distribution comparison
plt.subplot(2, 2, 2)
plt.hist(mc_results['no_hedge_pnl'], bins=30, alpha=0.3, label='No Hedging')
plt.hist(mc_results['daily_hedge_pnl'], bins=30, alpha=0.3, label='Daily Hedging')
plt.hist(mc_results['weekly_hedge_pnl'], bins=30, alpha=0.3, label='Weekly Hedging')
plt.hist(mc_results['monthly_hedge_pnl'], bins=30, alpha=0.3, label='Monthly Hedging')
plt.title('PnL Distribution by Hedging Strategy')
plt.xlabel('Profit/Loss')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)

#Boxplot comparison
plt.subplot(2, 2, 3)
plt.boxplot([mc_results['no_hedge_pnl'], 
             mc_results['daily_hedge_pnl'], 
             mc_results['weekly_hedge_pnl'], 
             mc_results['monthly_hedge_pnl']])
plt.xticks([1, 2, 3, 4], ['No Hedge', 'Daily', 'Weekly', 'Monthly'])
plt.title('PnL Distribution Comparison')
plt.ylabel('Profit/Loss')
plt.grid(True)

#Delta values along sample paths
plt.subplot(2, 2, 4)
for i in range(5):  #Plot delta for a few paths
    plt.plot(mc_results['delta_values'][i], alpha=0.7)
plt.title('Option Delta Evolution')
plt.xlabel('Trading Days')
plt.ylabel('Delta')
plt.grid(True)

plt.tight_layout()
plt.savefig('monte_carlo_hedging.png')
print("Created Monte Carlo hedging visualization")

#6. Reinforcement Learning for Optimal Hedging
class FXOptionEnvironment:
    def __init__(self, initial_price, strike, time_to_maturity, r_d, r_f, sigma, dt=1/252):
        self.initial_price = initial_price
        self.price = initial_price
        self.strike = strike
        self.time_to_maturity = time_to_maturity
        self.current_step = 0
        self.r_d = r_d
        self.r_f = r_f
        self.sigma = sigma
        self.dt = dt
        self.max_steps = int(self.time_to_maturity / self.dt)
        self.time_elapsed = 0
        self.option_price = black_scholes_merton_fx(self.price, self.strike, self.time_to_maturity, 
                                                   self.r_d, self.r_f, self.sigma)
        self.delta = compute_all_greeks(self.price, self.strike, self.time_to_maturity, 
                                       self.r_d, self.r_f, self.sigma)['delta']
        self.hedge_position = 0
        self.cash = 0
        self.transaction_cost = 0.0001  #1 bp transaction cost
        
    def reset(self):
        self.price = self.initial_price
        self.time_elapsed = 0
        self.option_price = black_scholes_merton_fx(self.price, self.strike, self.time_to_maturity, 
                                                    self.r_d, self.r_f, self.sigma)
        self.delta = compute_all_greeks(self.price, self.strike, self.time_to_maturity, 
                                       self.r_d, self.r_f, self.sigma)['delta']
        self.hedge_position = 0
        self.cash = 0
        self.current_step = 0
        return self._get_state()


    def _get_state(self):
        remaining_time = max(0.001, self.time_to_maturity - self.time_elapsed)
        
        bs_price = black_scholes_merton_fx(self.price, self.strike, remaining_time, 
                                        self.r_d, self.r_f, self.sigma)
        
        #Safe division
        price_diff = 0
        if abs(self.option_price) > 1e-6:  #Avoid division by very small numbers
            price_diff = (self.option_price - bs_price) / self.option_price
        
        return np.array([
            self.price / self.strike,
            remaining_time,
            self.delta,
            self.hedge_position,
            price_diff
        ])
    # def _get_state(self):
    #     remaining_time = max(0, self.time_to_maturity - self.time_elapsed)
    #     return np.array([
    #         self.price / self.strike,  # Normalized price
    #         remaining_time,
    #         self.delta,
    #         self.hedge_position,
    #         (self.option_price - black_scholes_merton_fx(self.price, self.strike, remaining_time, 
    #                                                     self.r_d, self.r_f, self.sigma)) / self.option_price
    #     ])
    
    def step(self, action):
        if self.current_step >= self.max_steps:
            return self._get_state(), 0, True, {
                "price": self.price,
                "option_price": self.option_price,
                "cash": self.cash,
                "hedge_position": self.hedge_position,
                "reason": "max_steps_reached"
            }
        #Action is the target delta position (0 to 1)
        action = max(0, min(1, action))  #Clip to [0, 1]
        
        #Calculate the difference between current and target position
        delta_change = action - self.hedge_position
        
        #Apply transaction cost for rebalancing
        transaction_cost = abs(delta_change) * self.price * self.transaction_cost
        self.cash -= transaction_cost
        
        #Update position
        cost_of_change = delta_change * self.price
        self.cash -= cost_of_change
        self.hedge_position = action
        
        #Simulate price movement
        old_price = self.price
        price_drift = (self.r_d - self.r_f - 0.5 * self.sigma**2) * self.dt
        price_diffusion = self.sigma * np.sqrt(self.dt) * np.random.normal()
        self.price *= np.exp(price_drift + price_diffusion)
        
        #Update time elapsed
        self.time_elapsed += self.dt
        
        #Calculate PnL from hedge position
        hedge_pnl = self.hedge_position * (self.price - old_price)
        self.cash += hedge_pnl
        
        #Calculate new option value
        remaining_time = max(0, self.time_to_maturity - self.time_elapsed)
        if remaining_time < 0.001:  #Option expiry
            payoff = max(0, self.price - self.strike)  #Call option payoff
            option_change = payoff - self.option_price
            self.option_price = payoff
            done = True
        else:
            new_option_price = black_scholes_merton_fx(self.price, self.strike, remaining_time, 
                                                      self.r_d, self.r_f, self.sigma)
            option_change = new_option_price - self.option_price
            self.option_price = new_option_price
            self.delta = compute_all_greeks(self.price, self.strike, remaining_time, 
                                           self.r_d, self.r_f, self.sigma)['delta']
            done = False
        
        #Calculate reward (minimizing tracking error)
        reward = -abs(hedge_pnl + option_change) - transaction_cost
        
        return self._get_state(), reward, done, {
            'price': self.price,
            'option_price': self.option_price,
            'delta': self.delta,
            'hedge_position': self.hedge_position,
            'cash': self.cash,
            'tracking_error': hedge_pnl + option_change
        }

#Create Deep Q-Learning agent for reinforcement learning
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  #discount rate
        self.epsilon = 1.0  #exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001))
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 2000:
            self.memory.pop(0)
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state.reshape(1, -1), verbose=0)[0])
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

#Example of using RL for hedging (just setup, not full training due to time constraints)
print("Setting up Reinforcement Learning for hedging...")
env = FXOptionEnvironment(initial_price=100, strike=100, 
                          time_to_maturity=1, r_d=0.05, r_f=0.02, 
                          sigma=0.2, dt=1/52)  #Weekly rebalancing

state_size = 5
action_size = 11  #Discretized delta positions from 0 to 1 in 0.1 increments
agent = DQNAgent(state_size=state_size, action_size=action_size)

#Function for fully training the agent (would take too long for live demo)
def train_rl_agent(episodes=1000):
    batch_size = 32
    rewards_history = []
    q_max_values = []
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_actions = []
        done = False
        
        while not done:
            action = agent.act(state)
            episode_actions.append(action)
            #Convert discrete action to continuous delta
            delta_action = action / (action_size - 1)
            next_state, reward, done, info = env.step(delta_action)
            
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            q_values = agent.model.predict(np.array([state]), verbose=0)
            q_max_values.append(np.max(q_values))
            
            agent.replay(batch_size)
        
        if e % 10 == 0:
            agent.update_target_model()
        
        rewards_history.append(total_reward)
        if e % 100 == 0:
            print(f"Episode: {e}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")
    
    return rewards_history

#For demo, simulate a few episodes to compare with traditional delta hedging
print("Comparing RL hedging to traditional delta hedging...")
num_test_episodes = 10
rl_results = []
delta_results = []

#Pre-train agent a little bit to get a sense of its behavior
agent.epsilon = 0.5  #Start with more exploitation than exploration
for _ in range(100):  #Limited pre-training for demo
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        delta_action = action / (action_size - 1)
        next_state, reward, done, info = env.step(delta_action)
        agent.memorize(state, action, reward, next_state, done)
        state = next_state
        if len(agent.memory) > 32:
            agent.replay(32)

agent.epsilon = 0.1  #Lower exploration for testing

for i in range(num_test_episodes):
    print(f"Testing episode {i+1}/{num_test_episodes}")
    #Test RL agent
    state = env.reset()
    rl_cash_history = [0]
    rl_hedge_history = [0]
    rl_price_history = [env.price]
    rl_option_history = [env.option_price]
    
    done = False
    while not done:
        action = agent.act(state)
        delta_action = action / (action_size - 1)
        next_state, reward, done, info = env.step(delta_action)
        state = next_state
        
        rl_cash_history.append(info['cash'])
        rl_hedge_history.append(info['hedge_position'])
        rl_price_history.append(info['price'])
        rl_option_history.append(info['option_price'])
    if i % 2 == 0:  #Print every other episode
        print(f"Completed {i+1}/{num_test_episodes} episodes")
    
    #Final PnL calculation
    rl_pnl = info['cash'] - rl_option_history[-1]
    rl_results.append({
        'pnl': rl_pnl,
        'cash': rl_cash_history,
        'hedge': rl_hedge_history,
        'price': rl_price_history,
        'option': rl_option_history
    })
    
    #Test traditional delta hedging
    env_delta = FXOptionEnvironment(initial_price=100, strike=100, 
                                   time_to_maturity=1, r_d=0.05, r_f=0.02, 
                                   sigma=0.2, dt=1/52)
    
    state = env_delta.reset()
    delta_cash_history = [0]
    delta_hedge_history = [0]
    delta_price_history = [env_delta.price]
    delta_option_history = [env_delta.option_price]
    
    done = False
    while not done:
        #Use the calculated delta as the action
        delta_action = env_delta.delta
        next_state, reward, done, info = env_delta.step(delta_action)
        state = next_state
        
        delta_cash_history.append(info['cash'])
        delta_hedge_history.append(info['hedge_position'])
        delta_price_history.append(info['price'])
        delta_option_history.append(info['option_price'])
    
    #Final PnL calculation
    delta_pnl = info['cash'] - delta_option_history[-1]
    delta_results.append({
        'pnl': delta_pnl,
        'cash': delta_cash_history,
        'hedge': delta_hedge_history,
        'price': delta_price_history,
        'option': delta_option_history
    })

#Visualize RL vs Delta Hedging
plt.figure(figsize=(16, 12))

#Plot performance comparison
plt.subplot(2, 2, 1)
plt.bar(['RL Hedging', 'Delta Hedging'], 
        [np.mean([r['pnl'] for r in rl_results]), 
         np.mean([r['pnl'] for r in delta_results])],
        yerr=[np.std([r['pnl'] for r in rl_results]), 
              np.std([r['pnl'] for r in delta_results])],
        alpha=0.7)
plt.title('Average PnL Comparison')
plt.ylabel('Profit/Loss')
plt.grid(True)

#Plot one example trajectory
example_idx = 0  #Pick the first simulation

plt.subplot(2, 2, 2)
plt.plot(rl_results[example_idx]['price'], label='FX Rate')
plt.axhline(y=100, color='r', linestyle='--', label='Strike Price')
plt.title('FX Rate Path')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(rl_results[example_idx]['hedge'], label='RL Hedge Ratio')
plt.plot(delta_results[example_idx]['hedge'], label='Delta Hedge Ratio')
plt.title('Hedging Strategy Comparison')
plt.xlabel('Time Steps')
plt.ylabel('Hedge Ratio')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(rl_results[example_idx]['cash'], label='RL Cash Balance')
plt.plot(delta_results[example_idx]['cash'], label='Delta Cash Balance')
plt.title('Cash Balance Evolution')
plt.xlabel('Time Steps')
plt.ylabel('Cash')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('rl_vs_delta_hedging.png')
print("Created RL vs Delta hedging visualization")

#Display all training results
print("\nImplementation Summary:")
print("1. Statistical Arbitrage: Pair trading with cointegration analysis")
print("2. Momentum Trading: Moving averages, RSI, and MACD with ensemble signals")
print("3. Order Book Analysis: Simulated order book with depth and market impact")
print("4. Full Greeks: Delta, Gamma, Vega, Theta and Rho with 3D visualization")
print("5. Monte Carlo: Hedging simulations with different rebalancing frequencies")
print("6. Reinforcement Learning: Deep Q-Learning for optimal hedging strategies")
print("\nAll visualizations saved to disk.")