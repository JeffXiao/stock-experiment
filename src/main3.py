#Import

import pandas as pd
import math, datetime
import time
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from matplotlib import style
from dateutil import parser

df = pd.read_csv('./SP500.csv')
df = df.set_index(df.Date)                              # indexing by date

# df = df.iloc[::-1]   # reverse all rows
# print(df.head())
# print(df.tail())
# df.plot(kind='box', subplots=True, layout=(1,5), sharex=False, sharey=False)
# plt.show()
# df.hist()
# plt.show()

df['OC_Change'] = (df['Close'] - df['Open']) / df['Open'] * 100
df['HL_Change'] = (df['High'] - df['Low']) / df['Low'] * 100
df = df[['Close', 'HL_Change', 'OC_Change']]

# print(df.tail())
forecast_col = 'Close'
forecast_out = 5                                        # shift 1% of rows upward
df = df.append(df.tail(forecast_out))                   # fake last forecast_day2 with dupication
df['Test'] = df[forecast_col].shift(-forecast_out)
# bk = df.tail(forecast_out)                            # make a backup copy
df.dropna(inplace=True)                                 # drop rows where label has Nan value
# print(df['Test'])

X = np.array(df.drop(['Test'], 1))                      # keep all columns except 'Test' in X
y = np.array(df['Test'])                                # keep column 'Test' in y

# print(len(X), len(y))

# df = df.append(bk)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
accuracy = clf.score(X_test, y_test)
print(accuracy)

X = X[:-forecast_out]
X_lately = X[-forecast_out:]
Forecast_set = clf.predict(X_lately)
print(Forecast_set)

df['Forecast'] = df[forecast_col].shift(-len(df))

df.at[df.iloc[-1].name, 'Forecast'] = value=df.iloc[len(df)-1]['Close']
epoch = datetime.datetime.utcfromtimestamp(0)
last_date = parser.parse(df.iloc[len(df) - 1].name)
next_unix = (last_date - epoch).total_seconds() + 86400 * 2
for i in Forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix).strftime('%Y-%m-%d')
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Date'] = pd.to_datetime(df.index)
plt.plot_date(x=df['Date'], y=df['Close'], linestyle='-')
plt.plot_date(x=df['Date'], y=df['Test'], linestyle='-')
plt.plot_date(x=df['Date'], y=df['Forecast'], linestyle='-')
plt.title('SP500 5 Days Forecast')
plt.legend(loc=0)
plt.xlabel('Date')
plt.ylabel('Price')

plt.show()
