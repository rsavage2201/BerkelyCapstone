## Project Big-5-BTC-Link
The objective of this project is to understand the connection and influence the current stock price of Apple, Microsoft, Meta, Google and Amazon (the “Big 5”) has on the price of BitCoin (BTC).

## Table of Contents
 - Description
 - Jupyter Notebook
 - Data
 - Results
 - Visualizations
 - Contact Information

## Description
This basics premise of this analysis is that the value of the “Big 5” technology companies is currently largely driven by their efforts in the AI field. As AI applications require significant computational resources, their value must be dependent on the cost of those computational resources, and and crypto currency prices are highly dependent on the overall cost of computation (“mining”).  So I wanted to determine if I could build a predictive model to capture this relationship.

## Data
In order to carry out this analysis I downloaded the stock prices of the tech companies from Yahoo Finance [YFinance repo](https://github.com/ranaroussi/yfinance), and the prices of the top 5 Cryptocurrencies from [CryptoCompare] (https://www.cryptocompare.com).
Both datasets were downloaded for the the two years between Jan 1st 2022 and Dec 31st 2023

## Data Cleaning
Lucking given the source and subject matter, the data was quite “clean” however after commencing the analysis, it made sense to focus solely on BTC  (rather than a basket of Crypto currencies), and I had to regularize the data and perform feature engineering to try to link current stock prices to future BTC prices

## Key Findings 
Although I was very optimistic a reliable predictive model could be developed, I’ve discovered that due to the volatility in BTC prices the models struggle significantly with predicting the future price of Bitcoin, especially over longer time horizons. Here’s a brief analysis:

    - Linear Regression performs reasonably well for very short-term predictions but the errors increase drastically for longer intervals. This indicates that Linear Regression is not capturing the complex, non-linear patterns in Bitcoin prices well over time.

    - ARIMA and SARIMA Models consistently have high RMSE values across all prediction intervals, showing they fail to adapt to the price movements of Bitcoin. These models might capture some patterns, but they struggle with the inherent volatility and non-stationarity of Bitcoin prices.

    - Random Forest Models (Grid Search and Bayesian Optimization) show moderate performance for short-term predictions but have very high errors for longer prediction intervals. This suggests that while they might capture some short-term trends, they lack the robustness needed for longer-term forecasts.

## Visualizations
The following Visualizations are included
    - Normalized Stock and Crypto prices excluding USDT and USDC
    - Model Performance RMSE

## Contact Information
You can contact me on (rowland_savage@yahoo.com)
