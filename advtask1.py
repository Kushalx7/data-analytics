import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

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

#creating target variable 
df['target'] = (df['daily_return_pct'] > 0).astype(int)
#now distributing test and train data
X = df[['close_price', 'volume', 'volatility_range', 'rsi_14', 'macd_value']]
y = df['target']
#train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#as ml models ned scaled data we used this and in this we used standard scalar as the output values are not bounded and it is done so all features contribute equally

#train models
lr = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

#evaluate models 

def evaluate(name, y_test, y_pred):
    print(name)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    print("---------------")

evaluate("Logistic Regression", y_test, lr.predict(X_test))
evaluate("Decision Tree", y_test, dt.predict(X_test))
evaluate("Random Forest", y_test, rf.predict(X_test))

#hyperparameter tuning:it means finding the best settings for your model to improve performance 
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)