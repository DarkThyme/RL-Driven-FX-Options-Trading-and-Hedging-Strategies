# RL-Driven-FX-Options-Trading-and-Hedging-Strategies


This project simulates and analyzes FX options trading strategies using advanced quantitative techniques. It includes a Reinforcement Learning (Deep Q-Learning) hedging model, cointegration-based pair trading, momentum indicators (MA, RSI, MACD), Greeks visualization, order book simulation, and Monte Carlo hedging—all implemented in Python and Tensorflow.

##  Features

- **Statistical Arbitrage**: Detects cointegrated currency pairs for pair trading
- **Momentum Trading**: Uses MA, RSI, and MACD with ensemble decision rules
- **Order Book Analysis**: Simulated depth and impact of trades
- **Option Greeks**: Computes and visualizes Delta, Gamma, Vega, Theta, and Rho in 3D
- **Monte Carlo Simulation**: Tests various hedging strategies across market scenarios
- **Reinforcement Learning**: Deep Q-Learning model to learn optimal hedging actions

## 📊 Visualizations
All visualizations (e.g., Greeks surface plots, RL vs Delta Hedging) are saved automatically in the output folder.

##  RL Details
- Framework: Tensorflow
- Training: CPU (Will take some time depending on the compute. Took 16 hours to train on i7 12th gen)
- Agent: Deep Q-Network (DQN)
- Episodes: 1000

## 🛠️ Tech Stack
- Python 3.12
- Tensorflow
- NumPy, Pandas, Seaborn, Matplotlib
- Scikit-learn
- Plotly

## You wiil need Alpha Vantage API Key for currency/forex data.

## 📁 How to Run
```bash
pip install -r requirements.txt

# Run the script
ML_Model.py
