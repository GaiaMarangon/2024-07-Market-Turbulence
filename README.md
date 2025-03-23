# 2024-07-Market-Turbulence

This project implements in ```MATLAB``` the Chow-Kritzman model, a tool for portfolio compositions built upon Markovitz optimal portfolio theory and accounting for market turbulence. 

# The model
The model provides a method for managing a portfolio of $N$ assets, optimally assigning a weight to each asset. Similar to Markovitz's theory, the weights are selected to maximize the returns while minimizing the risk (volatility). The weight are modified in time, based on historical data. To estimate mean and variance of the historical data, and therefore set the assets weights, the Chow-Kritzman model accounts for the state of the market, which can be found in two regimes: turbulent or non turbulent.
The probability of each regime can be estimated a priori (static Chow-Kritzman), or by describing the market as an Hidden Markov Model, with two market states - turbulent and non turbulent - each generating historical data with a different probability distribution (dynamic Chow-Kritzman).  
