# Ease of movement indicator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_eom(data):

    high = data['High']
    low = data['Low']
    vol = data['Volume']
    dm = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
    br = (vol / 100000000) / (high - low)

    eom = dm / br
    eom_ma = eom.rolling(14).mean()

    return eom_ma


data = pd.read_csv('TSLA.csv')
date = data['Date']

eom_i = calculate_eom(data)

plt.plot(date, eom_i, label='EOM', color='grey', alpha=0.7, linewidth=0.8)
zeros = np.zeros(len(data))
plt.plot(date, zeros, color='black', alpha=0.6)

plt.title('Ease of movement ( TSLA in FY 2022-23 )')
plt.ylabel('Price')
plt.xlabel('Timeline (Daily)')
plt.plot([], [], ' ', label='14 Period MA')
plt.plot([], [], ' ', label='Volume scale 10M')
plt.legend(loc='best')

ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])

plt.show()
