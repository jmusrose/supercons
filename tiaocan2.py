import numpy as np
import pandas as pd
import os
from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from time import time
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

df = pd.read_csv('CleanData.csv')
x = df.drop('critical_temp', axis = 1)
x_scale = sc.fit_transform(x)
x = pd.DataFrame(x_scale, columns = x.columns)
y = df['critical_temp']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)

rmse_list = []
r2_list = []
axis = range(3, 20, 1)
for i in axis:
    reg = XGBR(n_estimators=600, max_depth = i).fit(x_train,y_train)
    y_prediction = pd.Series(reg.predict(x_test))
    min_rmse = round(np.sqrt(mean_squared_error(y_test, y_prediction)), 4)
    r2 = round(r2_score(y_prediction, y_test), 4)
    rmse_list.append(min_rmse)
    r2_list.append(r2)
plt.figure(1, figsize=(6, 6))
plt.plot(axis, rmse_list, c='r', label = 'RMSE')
plt.plot(axis, r2_list, c='b', label = '$R^2$')
plt.legend()

plt.show()

