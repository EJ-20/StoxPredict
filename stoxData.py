import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping
from bs4 import BeautifulSoup
import requests
import re
import csv
from selenium import webdriver

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
import xml

#import stock data
# NVDA_stock = pd.read_csv('./NVDA.csv', index_col = "Date")
# NVDA_stock.head()
# AMD_stock =  pd.read_csv('./AMD.csv', index_col = "Date")
# AMD_stock.head()
# TSLA_stock =  pd.read_csv('./TSLA.csv', index_col = "Date")
# TSLA_stock.head()

PATTERN = re.compile('<.*?>')

urls = ['https://finance.yahoo.com/quote/NVDA/history?p=NVDA']
driver = webdriver.Firefox()



def scrapeData():

    driver.get(urls[0])
    driver.find_element_by_xpath('//a class="Fl(end) Mt(3px) Cur(p)" href="https://query1.finance.yahoo.com/v7/finance/download/NVDA?period1=1672063808&amp;period2=1703599808&amp;interval=1d&amp;events=history&amp;includeAdjustedClose=true" download="NVDA.csv"><svg class="Va(m)! Mend(5px) Stk($linkColor)! Fill($linkColor)! Cur(p)" width="15" style="fill:#0081f2;stroke:#0081f2;stroke-width:0;vertical-align:bottom" height="15" viewBox="0 0 24 24" data-icon="download"><path d="M21 20c.552 0 1 .448 1 1s-.448 1-1 1H3c-.552 0-1-.448-1-1s.448-1 1-1h18zM13 2v12.64l3.358-3.356c.375-.375.982-.375 1.357 0s.375.983 0 1.357L12 18l-5.715-5.36c-.375-.373-.375-.98 0-1.356.375-.375.983-.375 1.358 0L11 14.64V2h2z"></path></svg><span>Download</span></a>')
    page = requests.get(urls[0], headers={'User-Agent': 'Custom'})
    soup = BeautifulSoup(page.text, "lxml")

    table = soup.find('table', {"class":"W(100%) M(0)"})
    rows = []
    rows.append([])
    for i in table.findAll("th"):
        
        headers = i.find("span")
        #rows.append(re.sub(PATTERN, '', headers))
        rows[0].append(headers.contents[0])
    
    for i in table.findAll("tr"):
        
        data = i.findAll("td")
        if data != []: 
            rows.append([])
            for x in data:
                data2 = x.findAll("span")
                rows[-1].append(data2[0].contents[0])
    with open("stockPrice.csv", "w+") as my_csv:
        writer = csv.writer(my_csv)
        writer.writerows(rows)
    return rows


scrapeData()

NVDA_stock = pd.read_csv('./stockPrice.csv', index_col = "Date")
NVDA_stock.head()

#set values from imported files
x_dates_NVDA = [dt.datetime.strptime(d, "%Y-%m-%d").date() for d in NVDA_stock.index.values]

plt.figure(figsize=(15,10))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=20))





target_y = NVDA_stock['Close']
X_feat = NVDA_stock.iloc[:,0:3]

#Feature Scaling
sc = StandardScaler()
X_ft = sc.fit_transform(X_feat.values)
X_ft = pd.DataFrame(columns = X_feat.columns, data = X_ft, index= X_feat.index)

#Create windows 
def lstm_split(data, n_steps):
    X, y = [], []
    for i in range(len(data)-n_steps+1):
        X.append(data[i:i + n_steps, :-1])
        y.append(data[i+ n_steps-1, -1])


    return np.array(X), np.array(y)


X1, y1 = lstm_split(X_feat.values, 2)

train_split = 0.8
split_idx = int(np.ceil(len(X1) * train_split))
date_index = X_ft.index

X_train, X_test = X1[:split_idx], X1[split_idx:]
y_train, y_test = y1[:split_idx], y1[split_idx:]

X_train_date, X_test_date = date_index[:split_idx], date_index[split_idx:]

print(X1.shape, X_train.shape, X_test.shape, y_test.shape)


lstm = Sequential()
lstm.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu', return_sequences=True))
lstm.add(LSTM(50, activation="relu"))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer="adam")
lstm.summary()


history = lstm.fit(X_train, y_train, epochs=100, batch_size=4, verbose=2, shuffle=False)


y_pred = lstm.predict(X_test)


rmse = mean_squared_error(y_test, y_pred, squared=False)
mape = mean_absolute_percentage_error(y_test, y_pred)
print("RSME: ", rmse)
print("MAPE: ", mape)




#plot data
plt.plot(x_dates_NVDA, NVDA_stock["High"], label="High")
plt.plot(x_dates_NVDA, NVDA_stock["Low"], label="Low")
plt.xlabel("Time Scale")
plt.ylabel("Scaled USD")
plt.legend()
plt.gcf().autofmt_xdate()
plt.show()