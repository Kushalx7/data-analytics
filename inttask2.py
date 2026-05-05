import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
df= pd.read_csv("D:/trend analysis/Market_Trend_Analysis.csv")
print(df.head())
print(df.columns)
print(df.describe())
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())

df.columns=df.columns.str.lower()
df.columns=df.columns.str.strip()
print(df.columns)

#as my date colum is currently in string i will convert to date time 
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df.set_index('date', inplace=True)
print(df.info())
#plotting time series
df['close_price'].plot(figsize=(10,5))
plt.title("Close Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

#moving average
df[['close_price', 'sma_50']].plot(figsize=(10,5))
plt.title("Close Price with 50-Day Moving Average")
plt.show()
#in this using raw data we smooth it by taking avrage of recent values
#decomposing time series
result = seasonal_decompose(df['close_price'], model='additive', period=30)
result.plot()
plt.show()
#in this we break data into parts like trend,seasonality and residual(noise)
#trend: long-term direction
#seasonality: repeating patterns
#residual: random behavior 