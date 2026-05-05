import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

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
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df.set_index('date', inplace=True)
print(df.info())

#K-means use only numerical columns
X = df[['close_price', 'volume', 'daily_return_pct', 'volatility_range']]

#standardize the data as clustering depends on distance = scale matters
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#using elbow emthod finding the best nuber of clusters
wcss = []  # within-cluster sum of squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()
#from elbow method we can see the best numbers of clusters is 3 as after that it bends down

#applying kmeans with 3 clusters 
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['cluster'] = clusters
#visualzing clusters
plt.scatter(df['close_price'], df['volume'], c=df['cluster'], cmap='viridis')
plt.xlabel("Close Price")
plt.ylabel("Volume")
plt.title("K-Means Clusters")
plt.show()

sns.scatterplot(x='close_price', y='volume', hue='cluster', data=df)
plt.title("Cluster Visualization")
plt.show()