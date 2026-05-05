import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("D:/10thsem Visualiztion/Agrofood_co2_emission.csv/Agrofood_co2_emission.csv")
print(df.head())
print(df.columns)
print(df.info())
print(df.describe())
print(df.isnull().sum())
df=df.fillna(df.mean(numeric_only=True))
#putting value 0 in null value mmeans the emission is 0 so rather than that i took the mean and put values of all in it
print(df.isnull().sum())
df.columns = df.columns.str.lower().str.strip() \
    .str.replace(' ', '_', regex=False) \
    .str.replace('-', '_', regex=False) \
    .str.replace('[^a-z0-9_]', '', regex=True) \
    .str.replace('_+', '_', regex=True)
#Bar plot(Categorical data)
top_area = df.groupby('area')['total_emission'].sum().sort_values(ascending=False).head(10)

top_area.plot(kind='bar')
plt.title("Top 10 Areas by Total Emission")
plt.xlabel("Area")
plt.ylabel("Total Emission")
plt.xticks(rotation=45)
plt.show()

#line chart(trends over time)
yearly = df.groupby('year')['total_emission'].mean()

plt.plot(yearly.index, yearly.values)
plt.title("Average Emission Over Years")
plt.xlabel("Year")
plt.ylabel("Average Emission")
plt.show()

#scatter plot(relationship)
sns.scatterplot(x='total_emission', y='average_temperature_c', data=df)
plt.title("Emission vs Temperature")
plt.show()