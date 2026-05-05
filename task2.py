#in the task 1 dataset mostly my data was in category so it was difficult to find mean.mode and others so now i am using naother dataset 
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

#now standardixing column names
df.columns = df.columns.str.lower().str.strip() \
    .str.replace(' ', '_', regex=False) \
    .str.replace('-', '_', regex=False) \
    .str.replace('[^a-z0-9_]', '', regex=True) \
    .str.replace('_+', '_', regex=True)
#when we use regexFalse it treat the pattern as plain text,not as a regular expression
# regex true match patterns(rules),more powerful while false match exact text only
print(df.columns)

#mean
print(df.mean(numeric_only=True))
#median
print(df.median(numeric_only=True))
#mode
print(df.mode(numeric_only=True))
#standadr deviation
print(df.std(numeric_only=True))

#also df.describe()gives allthese values in one go
print(df.describe())

#Vizualization
#Histogram (is shows how values are spread)
df['total_emission'].hist()
plt.title("Distribution of Total Emission")
plt.xlabel("Total Emission")
plt.ylabel("Frequency")
plt.show()

#Boxplot(outliers detection)
df.boxplot(column='total_emission')
plt.title("Boxplot of Total Emission")
plt.show()

#Scatterplot(relationship between two variables)
plt.scatter(df['total_emission'], df['average_temperature_c'])
plt.xlabel("Total Emission")
plt.ylabel("Average Temperature")
plt.title("Emission vs Temperature")
plt.show()
# Calculate correlation
corr = df.corr(numeric_only=True)
print(corr)

#heatmam pf correlation
corr = df.corr(numeric_only=True)

plt.imshow(corr, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix")
plt.show()
