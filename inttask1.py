import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
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

#let's seperate features and target
X=X = X = df[['total_emission', 'year']]
y=df['average_temperature_c']

#now train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 )
#test size 0.2 means 20% data will be used for testing and 80% for training
#random_state = 42 means same split every time when we run code
#now training model
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
#now making predictions 
y_pred = model.predict(X_test)

#eavluating model 

# r squared
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)
#mse 
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)

#intercept = when all input variables are 0,temperature would be -67.168
#coefficient1 = total emision has almost no effect on temp
#coefficient2 = each year temp increase by 0.0339 every year 