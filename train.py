'''linear regression'''
import numpy as np

import pandas as pd
dataset = pd.read_csv("D:\ML\HEROKU\linear-regression01\Salary_Data.csv")
X = dataset.iloc[:, :-1].values
print(X)
y = dataset.iloc[:, -1].values
print(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(regressor.predict([[3]]))
import joblib
joblib.dump(regressor,'regressor_joblib.sav')
