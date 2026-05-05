#Importing necessary libraries and loaidng the dataset
import pandas as pd

df = pd.read_csv("D:/kagglee/apple_jobs.csv")
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())

#checking the null values
print(df.isnull())
print(df.isnull().sum())
#in this way we can check null values the first statement shouws null values as true and false of all rows and columns while adding .sum() it sum each cloumn and gives null values 
missing_percent = (df.isnull().sum() / len(df)) * 100
print(missing_percent)
#shows missing values in percentage of each column 
#not required but just incase for better understanding

print(df.columns)
#when i analyze the column i found column education&experience which is seen very lenghty and not appropriate for a column so will rename it
df.rename(columns={'education&experience': 'edu_exp'}, inplace=True)
#when inplace=True is used it will change the column name in original dataframe and if used false it will return a new dataframe with the changed column 
#handling missing values
df['preferred_qual'] = df['preferred_qual'].fillna('Not Specified')

df['edu_exp'] = df['edu_exp'].fillna('Not Specified')
#additionaly i can drop the rows with misisng values or can put value 0 if it is in integer 

#verifying cleaning
print(df.isnull().sum())

#the expected output should show 0 null values in all columns
# now removing duplicates if any
print(df.duplicated().sum())
#no duplicates were found in the dataset
#if any duolicates were found we can remove using df =df.drop_duplicates()
# standardizing the data :it means making dataset consistent in format,strcuture and represntation
df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('&', '_')
#this code helps cleaning the column names so they are lowercase,space free and code-friendly
print(df.columns)