import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import math 
#from datetime import datetime

df = pd.read_csv('Train.csv')
print(df.head())
df=df.head(300)

cols = list(df)[2:19]
print(cols)

df_for_tra = df[cols].astype(float)

scaler = StandardScaler()
scaler = scaler.fit(df_for_tra)
df_for_tra_scaled = scaler.transform(df_for_tra)

trainX = []
trainY = []

n_future = 1   # Number of days we want to look into the future based on the past days.
n_past = 14 # Number of past days we want to use to predict the future.

for i in range(n_past, len(df_for_tra_scaled) - n_future +1):
    trainX.append(df_for_tra_scaled[i - n_past:i, 0:df_for_tra.shape[1]])
    trainY.append(df_for_tra_scaled[i + n_future - 1:i + n_future, 9])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

df_for_tra_scaled

trainY

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True)) 
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(trainX, trainY, epochs=5, batch_size=3, validation_split=0.1, verbose=1,)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

df2= df.head(300)
train_dates = pd.to_datetime(df2['Date'])
print(train_dates.tail(15))

n_past = 300
n_days_for_prediction=286 
#let us predict past 15 days

predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_days_for_prediction, freq='M').tolist()
print(predict_period_dates)

prediction = model.predict(trainX[-n_days_for_prediction:])

prediction_copies = np.repeat(prediction, df_for_tra.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(prediction_copies)[:,9]

forecast_dates = []
for time_i in predict_period_dates:
    forecast_dates.append(time_i.date())
    
df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'price':y_pred_future})
df_forecast['Date']=pd.to_datetime(df_forecast['Date'])

original = df[['Date', 'price']]

df_forecast

original.plot(x='Date',y='price')
original= pd.concat([original,df_forecast],axis=0)
original.plot(x='Date',y='price')
#f_forecast.plot(x='Date',y='price')

Diff = df['price']-df_forecast['price']
Diff

RMSE = math.sqrt((Diff*Diff).sum())/len(Diff)
print(RMSE)

RMSE_PER = np.sqrt(np.mean(np.square(((df['price'] - df_forecast['price']) / original['price'])), axis=0))
print("Efficiency = "+str(100-RMSE_PER*100)+" %")



