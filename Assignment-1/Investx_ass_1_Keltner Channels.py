# Import pandas and numpy libraries
import pandas as pd
import numpy as np

# Define a function to calculate the Keltner Channels
def KC(data, n=20, m=2):
  # Calculate the typical price for each day
  data['Typical Price'] = (data['High'] + data['Low'] + data['Close']) / 3
  # Calculate the EMA of the typical price with a lookback period of n
  data['TP_EMA'] = data['Typical Price'].ewm(span=n, adjust=False).mean()
  # Calculate the true range for each day
  data['True Range'] = np.maximum(data['High'] - data['Low'], np.abs(data['High'] - data['Close'].shift(1)), np.abs(data['Low'] - data['Close'].shift(1)))
  # Calculate the EMA of the true range with a lookback period of n
  data['TR_EMA'] = data['True Range'].ewm(span=n, adjust=False).mean()
  # Calculate the upper band by adding m times the EMA of the true range to the EMA of the typical price
  data['Upper Band'] = data['TP_EMA'] + m * data['TR_EMA']
  # Calculate the lower band by subtracting m times the EMA of the true range from the EMA of the typical price
  data['Lower Band'] = data['TP_EMA'] - m * data['TR_EMA']
  # Return the data with the Keltner Channels columns
  return data

# Test the function with an example
# Read the sample data from a csv file
data = pd.read_csv('TSLA.csv')
# Apply the KC function with a lookback period of n=20 and a multiplier of m=2
data = KC(data, n=20, m=2)
# Print the first 10 rows of the data
print(data.head(10))
