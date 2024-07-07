# Ichimoku Cloud Lines 
import pandas as pd
import matplotlib.pyplot as plt

def calculate_tenkan_sen(high, low):
    # Conversion line
    return (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2

def calculate_kijun_sen(high, low):
    # Base line
    return (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2

def calculate_senkou_span_a(tenkan_sen, kijun_sen):
    # Leading Span A
    return ((tenkan_sen + kijun_sen) / 2).shift(26)

def calculate_senkou_span_b(high, low):
    # Leading Span B
    return (high.rolling(window=52).max() + low.rolling(window=52).min() / 2).shift(26)

def calculate_chikou_span(close):
    # Lagging Span
    return close.shift(-26)

# Usage
data = pd.read_csv('TSLA.csv')  
high = data['High']
low = data['Low']
close = data['Close']
date=data['Date']

tenkan_sen = calculate_tenkan_sen(high, low)
kijun_sen = calculate_kijun_sen(high, low)
senkou_span_a = calculate_senkou_span_a(tenkan_sen, kijun_sen)
senkou_span_b = calculate_senkou_span_b(high, low)
chikou_span = calculate_chikou_span(close)

# Print the calculated values
print("Tenkan-sen:", tenkan_sen.tail())
print("Kijun-sen:", kijun_sen.tail())
print("Senkou Span A:", senkou_span_a.tail())
print("Senkou Span B:", senkou_span_b.tail())
print("Chikou Span:", chikou_span.tail())

plt.plot(date, tenkan_sen, label='Conversion Line (9)', color='blue', alpha=0.6, linewidth=0.7)
plt.plot(date, kijun_sen, label='Base Line (26)', color='black', alpha=0.6, linewidth=0.7)
plt.plot(date, senkou_span_a, label='Leading Span A (26)', color='green', alpha=0.3, linewidth=0.7)
plt.plot(date, senkou_span_b, label='Leading Span B (52)', color='red', alpha=0.3, linewidth=0.7)
plt.fill_between(date, senkou_span_a, senkou_span_b, color='grey', alpha=0.15)
plt.plot(date, chikou_span, label='Lagging Span (-26)', color='grey', alpha=0.6, linewidth=0.6)

plt.title('Ichimoku Cloud ( TSLA in FY 2022-23 )')
plt.ylabel('Price')
plt.xlabel('Timeline (Daily)')
plt.legend(loc='best')

ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])

plt.show()
