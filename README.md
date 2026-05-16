# Ex.No: 6               HOLT WINTERS METHOD
### Date: 16-5-2026
### AIM:
To implement the Holt Winters Method Model using Python.
### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset, perform data exploration
data = pd.read_csv('/content/Balaji Fast Food Sales.csv')
# Explicitly convert the 'date' column to datetime objects and set it as the index
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data.dropna(subset=['date'], inplace=True) # Drop rows where date parsing failed
data.set_index('date', inplace=True)
data.head()

# Resample and plot - selecting 'transaction_amount' for time series analysis
data_monthly = data['transaction_amount'].resample('MS').sum()
data_monthly.head()
data_monthly.plot()
plt.title('Monthly Sales Transaction Amount')
plt.show()

# Scale the data
scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),
                        index=data_monthly.index)
scaled_data.plot()
plt.title('Scaled Data')
plt.show()

# Define seasonal period for monthly data
seasonal_period = 12

# Check for seasonality - decompose
# seasonal_decompose requires at least two full cycles of data
if len(data_monthly) >= 2 * seasonal_period:
    decomposition = seasonal_decompose(data_monthly, model='additive', period=seasonal_period)
    decomposition.plot()
    plt.show()
else:
    print(f"Skipping seasonal decomposition: Not enough data for 2 cycles of {seasonal_period}-period seasonality.")
    print(f"Requires at least {2 * seasonal_period} observations, but only {len(data_monthly)} are available.")

# multiplicative seasonality can't handle non-positive values
# The current data_monthly now represents 'transaction_amount', which should always be positive.
# This line is kept as per the original notebook logic for scaled_data.
scaled_data = scaled_data + 1

# Split train and test data
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data  = scaled_data[int(len(scaled_data) * 0.8):]

# Create and train Holt-Winters model
# Determine if a seasonal model can be used based on data length
if len(train_data) < 2 * seasonal_period:
    print(f"Warning: Training data (len={len(train_data)}) is too short for a seasonal Holt-Winters model with period={seasonal_period}. Using non-seasonal model (seasonal=None).")
    seasonal_param_train = None
else:
    seasonal_param_train = 'mul'

model_add = ExponentialSmoothing(train_data, trend='add', seasonal=seasonal_param_train, seasonal_periods=seasonal_period).fit()
test_predictions_add = model_add.forecast(steps=len(test_data))

# TEST PREDICTION plot
ax = train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(['train_data', 'test_predictions_add', 'test_data'])
ax.set_title('Visual Evaluation')
plt.show()

# Model performance metrics
rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print(f'RMSE: {rmse}')
print(f'Standard Deviation: {scaled_data.std()}, Mean: {scaled_data.mean()}')

# Final model on full data and predict future
if len(data_monthly) < 2 * seasonal_period:
    print(f"Warning: Full data (len={len(data_monthly)}) is too short for a seasonal Holt-Winters model with period={seasonal_period}. Using non-seasonal model (seasonal=None) for final prediction.")
    seasonal_param_full = None
else:
    seasonal_param_full = 'mul'

final_model = ExponentialSmoothing(data_monthly, trend='add', seasonal=seasonal_param_full,
                                   seasonal_periods=seasonal_period).fit()
final_predictions = final_model.forecast(steps=int(len(data_monthly) / 4))

# FINAL PREDICTION plot
ax = data_monthly.plot()
final_predictions.plot(ax=ax)
ax.legend(['data_monthly', 'final_predictions'])
ax.set_xlabel('Months')
ax.set_ylabel('Monthly Sales Transaction Amount') # Updated ylabel
ax.set_title('Prediction')
plt.show()
```
### OUTPUT:

<img width="633" height="511" alt="Screenshot 2026-05-16 094243" src="https://github.com/user-attachments/assets/73a613b1-3e03-4a64-827c-82aeb6e34486" />

<img width="1062" height="527" alt="Screenshot 2026-05-16 094459" src="https://github.com/user-attachments/assets/9415c916-95c1-4d7b-ace1-ff994852031b" />


#### TEST_PREDICTION:

<img width="1188" height="527" alt="image" src="https://github.com/user-attachments/assets/e3df1aa2-4725-4de4-917d-958c28bbeb30" />


#### FINAL_PREDICTION:


<img width="516" height="472" alt="Screenshot 2026-05-16 094733" src="https://github.com/user-attachments/assets/884be171-d5ed-4e80-a960-906b626051f5" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
