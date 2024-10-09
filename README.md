**BTC Price Prediction Project**

**Overview**
This project intends to explore if there is a predictive relationship between the current stock price of the “Big 5” technology companies (Google, Amazon, Meta, Microsoft, and Apple), and the future price of the major crypto currencies, refined to focus solely on Bitcoin (BTC).

If this relationship exists, we want to determine if it can be accurately modelled it using the techniques covered during this course, with the goal to evaluate whether a model can be created that is accurate enough to guide profitable trading decisions.

**Business Understanding**
The thesis is that a significant portion of the top 5 technology company’s (Google, Amazon, Meta, Microsoft, and Apple) future revenue and growth - and therefore value, is heavily dependent on ML/AI. This value will be reflected in their current Stock Price.

The value of ML/AI in turn is highly dependent on the raw cost of computation, with which the price of the major crypto currencies are highly correlated with. One “mines” for Bitcoin for example, by using CPUs/GPUs to solve complex mathematical problems (hashes) to validate and add transactions to the blockchain. So ones of the reasons Bitcoins price is so volatile is because the underlying cost of computation changes (e.g. the recent introduction of the Nvidia GPU chips).

So, there are two different financial instruments, valued (separately) on the same fundamentals, and by utilizing multiple machine learning and time-series models, this project aims to build a predictive framework that can be used for a trading strategy.

**Table of Contents**
1.	Installation
2.	Usage
3.	Module Description
4.	Modeling Approach
5.	Modelling Evaluation
6.	Implementation of Best Models
7.	Dependencies
8.	Findings and Recommendations
9.	Output Files and Images
10.	License
11.	Contact

**Installation**
To run this project, follow these steps:
1.	Access my [Capstone's public GitHub repo] (https://github.com/rsavage2201/BerkelyCapstone)
2.	Using Jupyter Notebook, run the Python Notebook RMS Capstone 100824 (Final).ipynb

**Usage**
To execute each module, run the scripts sequentially. Each module builds upon the outputs of the previous ones, so they must be executed in order. Each module generates specific outputs such as datasets, trained models, visualizations, and evaluation reports.

**Module Description**
1.	Module 1: Stock Price Data Download.
-	Download and visualize the historical adjusted closing prices for the "Big 5" technology companies over a fixed period (from January 1, 2022, to December 30, 2023).
-	Output: A CSV file (combined_stock_data.csv) containing the cleaned and merged stock prices data for the five companies.
2.	Module 2: Crypto Currency Data Download
-	Downloads historical cryptocurrency price data for specified assets for the date range and apply additional transformations to align with the stock data analysis.
-	Output: The module generates a CSV file (crypto_data_combined.csv) containing the historical closing prices for each cryptocurrency.
3.	Module 3: Initial Exploration of the data
-	Explore, visualize, and analyze the combined stock and cryptocurrency data by plotting price trends, normalizing values for comparison, and calculating correlations and volatility.
-	Output: Several plots to visualize the above.
4.	Module 4: Feature Engineering and Data Preparation
-	Performs feature engineering, scaling, and preparation of stock and cryptocurrency data, particularly BTC, for predictive modeling by creating lag features, rolling averages, EMAs, and volatility measures.
-	Output: A combined and transformed dataset saved as combined_data.csv, containing engineered features, future BTC prices as targets, and scaled values for modeling. 
5.	Module 5: Model Creation and Training 
-	Trains multiple machine learning and time series models (Linear Regression, Random Forest, ARIMA, SARIMA) to predict Bitcoin prices at different time intervals (0, 1, 7, 14, and 28 days ahead). Evaluate these models using RMSE and R² metrics to determine their performance across different forecast horizons.
-	Output: A summary CSV file (model_performance_summary.csv) containing the RMSE and R² values for each model and each forecast interval. 
6.	Module 6: Model Evaluation and Visualization
-	Visualizes and analyzes the performance of different models in predicting BTC prices across various time intervals by comparing RMSE and R² values.
-	Output: Bar plots and heatmaps for RMSE and R² values across models and intervals, saves these visualizations as PNG files, and outputs a summary CSV file (model_performance_summary_averages.csv) showing the average performance metrics for each model.
7.	Module 7: Implementing LSTM Model (Added)
-	Evaluate various LSTM neural network configurations for predicting BTC prices at different time intervals.
-	Output: A CSV file (lstm_config_comparison.csv) with RMSE and R² values for each LSTM configuration and target interval, alongside visualizations of RMSE and R² comparisons saved as PNG files.
8.	Module 8: Evaluate Models Predictive ability
-	Predicts Bitcoin prices for 2024 using SARIMA and Random Forest models, based on the performance of the selected technology stock prices and previous BTC prices. These predictions are compared with the actual Bitcoin prices for 2024.
-	Output: A CSV file (btc_predictions_deltas.csv) containing actual BTC prices, predictions from both models, and their deltas. It also creates visualizations comparing actual BTC prices against predictions from each model and a combined plot.
9.	Module 9: Evaluate Trading Strategy Based on Models
-	Evaluates trading strategies based on SARIMA and Random Forest model predictions by simulating buy-and-sell actions of Bitcoin with a 7-day horizon and comparing profits or losses generated by each model given the actual Bitcoin prices.
-	Output: A summary of trading performance for each model, including total amounts spent, realized gains, and percentage gain/loss, as well as the starting and ending Bitcoin prices with percentage change over the period.

**Modeling Approach**
Models Implemented:
1.	Linear Regression: A baseline model to establish a simple linear relationship between the features and the target (BTC prices at various intervals). It’s straightforward and helps to set a benchmark.
2.	ARIMA: A time series model used for forecasting. Auto ARIMA automatically determines the optimal parameters (p, d, q) based on the historical BTC prices. This model was selected due to its strength in short-term time series forecasting.
3.	SARIMA: An extension of ARIMA that incorporates seasonality (SARIMAX). Seasonal parameters are set to account for patterns in the BTC price data, specifically using a seasonal period of 7 (weekly cycles), to model potential weekly price movements.
4.	Random Forest: A tree-based ensemble model suitable for capturing complex non-linear relationships between features and BTC price movements. We trained a Random Forest model to handle each of the forecast intervals separately.

Forecast Intervals:
Models were evaluated the for predictions at the following intervals:
1.	BTC_0d: Predicting the current BTC price (used as a baseline).
2.	BTC_1d: Predicting 1 day ahead.
3.	BTC_7d: Predicting 7 days ahead.
4.	BTC_14d: Predicting 14 days ahead.
5.	BTC_28d: Predicting 28 days ahead.
These intervals were chosen to assess the models’ performance in both short-term (daily) and longer-term (weekly and monthly) forecasts.

Random Forest and Linear Regression Implementation:
For each forecast interval (e.g., BTC_1d, BTC_7d), a separate model was trained. This allowed the models to focus on the specifics of each interval, such as different patterns or feature importance relevant to that time frame.

ARIMA and SARIMA Parameter Selection:
ARIMA: Auto ARIMA was used to optimize the ARIMA parameters (p, d, q). This automated process ensures that the most suitable configuration is selected based on the data for each interval.
SARIMA: SARIMA used the same non-seasonal parameters identified by ARIMA but added seasonal parameters (1, 1, 1, 7) to account for any cyclical patterns within the weekly timeframe.

LSTM Model Tuning
In addition to the above models, I also experimented with Long Short-Term Memory (LSTM) networks, given their effectiveness in time series predictions. Here's how the LSTM models were designed and tuned:

Tuning Hyperparameters:
Multiple configurations with varying numbers of LSTM layers, dropout rates, and units per layer were designed:
-	Baseline: A single LSTM layer with 50 units.
-	Configuration 1: Two LSTM layers (50 units each) with dropout rates of 0.2 after each layer to prevent overfitting.
-	Configuration 2: A single LSTM layer with 100 units and a dropout rate of 0.3. The activation function used was tanh, and the optimizer was set to rmsprop.
-	Configuration 3: Three LSTM layers (100, 50, 25 units) with dropout applied to the first two layers (0.2 each). The relu activation function and adam optimizer were chosen for this configuration.
Batch Size and Epochs:
-	Different batch sizes (16, 32, 64) and epochs (100 or 150) were used for each configuration to test model performance and speed up convergence.
Early Stopping:
-	An EarlyStopping callback was applied during training to halt the process if the model's validation loss did not improve for 10 consecutive epochs, preventing overfitting and saving time.
This module, therefore, offers a comprehensive evaluation of various models and configurations, exploring both traditional ML and neural network approaches across multiple time horizons.

Model Evaluation
Objective Overview: Modules 6 and 7 are designed to evaluate the performance of various models (Linear Regression, ARIMA, SARIMA, Random Forest, and LSTM) for predicting Bitcoin (BTC) prices across multiple time horizons: 0, 1, 7, 14, and 28 days into the future. The results are assessed using Root Mean Squared Error (RMSE) and R² metrics.

1.	Linear Regression:
-	Performed well for shorter-term predictions (e.g., BTC_0d and BTC_1d) with high R² values (1.0 and 0.97, respectively).
-	Performance deteriorated significantly for longer intervals (e.g., BTC_28d), showing negative R² values and high RMSE, indicating that the model struggles with long-term predictions.
2.	ARIMA and SARIMA:
-	ARIMA generally performed poorly across all intervals, with negative R² values and high RMSE scores, suggesting that its capability to capture the BTC price dynamics is limited.
-	SARIMA showed a slight improvement over ARIMA, particularly for short-term intervals (BTC_0d and BTC_1d), where it achieved positive R² values. However, its performance declined for longer intervals (e.g., BTC_28d), where the R² values were negative and the RMSE remained high, indicating poor predictive accuracy as the forecast horizon increased.
3.	Random Forest:
-	Outperformed other traditional models (e.g., ARIMA, SARIMA) with positive R² values for short- and medium-term predictions (BTC_0d, BTC_1d, BTC_7d, and BTC_14d). The model’s R² values decreased and even turned negative for BTC_28d, demonstrating reduced reliability for long-term forecasting.
-	Achieved the lowest RMSE values for short-term predictions, highlighting its effectiveness in capturing the immediate patterns in BTC price fluctuations.
4.	LSTM Models:
-	Multiple configurations of LSTM models were tested (Baseline, Configuration 1, Configuration 2, Configuration 3), experimenting with different hyperparameters like the number of layers, units per layer, activation functions, dropout rates, and optimizers to optimize performance:
-	Baseline: A simple LSTM with one layer, which generally showed very high RMSE and negative R² values across all intervals.
-	Configuration 1: Included two LSTM layers and dropout; although some improvements were observed, it still showed poor performance, particularly for the short-term intervals (BTC_0d and BTC_1d).
-	Configuration 2: Tuned for a single large LSTM layer with a higher dropout rate and the tanhactivation function. However, this configuration yielded high RMSE and negative R² values, indicating it was not effective for this task.
-	Configuration 3: A deeper network with three LSTM layers and varying dropout rates showed the best results among LSTM models, particularly for BTC_7d. Despite this, RMSE remained high, and R² values were still negative, demonstrating the challenge of tuning LSTM models effectively for this data.

**Implementation of Best Models**
Module 8 implements and simulates a trading strategy using the predictions from the SARIMA and Random Forest models to evaluate how well these models can forecast BTC price movements. The strategy involves buying BTC when the predicted price (from either model) is higher than the actual price and selling it 7 days later.

Output Summary:
- The module produced plots (combined_vs_actual_btc_prices.png) and a detailed table with the actual BTC prices, model predictions (SARIMA and Random Forest), and the difference (delta) between the actual prices and the predictions for each day.
- This output helps visualize and assess the accuracy of the models' predictions and their alignment with actual BTC price changes. The deltas reveal whether the models tend to overestimate or underestimate BTC prices at different points, which is critical for evaluating the effectiveness of the trading strategy.

Module 9 builds on Module 8 by calculating the total financial impact of executing the trading strategy based on the model predictions over the 9-month period from January 1, 2024, to September 30, 2024.

Output Summary:
- The BTC market saw a 31.69% increase in price from $49,958.22 at the start of the year to $65,790.66 at the end of September 2024.
- SARIMA Model:
-	Total spent: $5,485,166.23
-	Total realized: $5,550,048.20
-	Net gain: $64,881.97 (1.18% gain)
-	Random Forest Model:
-	Total spent: $4,165,978.86
-	Total realized: $4,349,286.58
-	Net gain: $183,307.72 (4.40% gain)

**Findings**
1.	Model Performance:
-	The Random Forest model outperformed other models with a percentage gain of 4.40%, demonstrating better responsiveness to BTC price patterns.
-	The SARIMA model achieved a 1.18% gain, indicating its utility but showing room for improvement.
-	Short-term predictions (0 and 1 day ahead) are most effectively handled by the Linear Regression and Random Forest models, with both showing low RMSE and high R² values.
-	Medium-term predictions (7 and 14 days) have reasonable accuracy with the Random Forest model, but performance diminishes as the interval increases.
-	Long-term predictions (28 days) show a significant decline across all models, with negative R² values indicating that these models cannot reliably predict BTC prices over a 28-day horizon.
-	LSTM configurations tested did not outperform traditional models, even with extensive hyperparameter tuning, likely due to the complexity and volatility of BTC prices that require more advanced architectures or feature engineering.

2.	Trading Simulation:
-	The trading strategy based on Random Forest predictions resulted in a higher profit than the SARIMA-based strategy, however, neither produced a return that seemed commiserate with the risk being incurred, and neither outperformed a simple “Buy and Hold” strategy.
-   Ultimately the project failed to create a model accurate enough to guide profitable trading decisions, but this is a very hard problem to solve and this project is a great starting point. 

**Recommendations**
1.	Expand Features:
-	Incorporate other relevant financial data, such as currency exchange rates, global market indices, or sentiment analysis from news and social media.
2.	Model Improvement:
-	Enhance the SARIMA model by exploring hybrid models (e.g., combining SARIMA with LSTM) to better capture BTC price volatility.
-	Focus on short-term forecasting: Models like Random Forest and Linear Regression demonstrate strong capabilities for short-term BTC predictions and should be preferred for intervals of up to 7 days.
-	Further tuning for LSTM models: Given their potential, additional configurations (e.g., deeper networks, different input sequences, or other recurrent architectures like GRU) may be explored to improve performance.
3.	Explore Other Time Horizons
-	The models work better with shorter time horizons, my next step is to explore the 48 hour time horizon and repeat the trading experiment.
4.	Investment Strategy:
-	Consider implementing an ensemble approach that combines the top-performing models for a more robust and diversified trading strategy.

**Output Files and Images**
1.	combined_stock_data.csv
2.  crypto_data_combined.csv
3.	normalized_prices_excluding_usdc_usdt.png
4.	model_performance_summary.csv
5.	average_rmse_per_model.png
6.	average_r2_per_model.png
7.	lstm_config_comparison.csv
8.	combined_vs_actual_btc_prices.png
9.	trading_strategy_results.txt

**Dependencies**
Ensure the following libraries are installed:
- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels, pmdarima, yfinance, cryptocompare, joblib, shap, datetime

This project is licensed under the MIT License. See the LICENSE file for more details.

**Contact**
For questions or feedback: Rowland Savage
Email: Rowland_Savage@yahoo.com
GitHub: [Rowland's Repo](https://github.com/rsavage2201)
