#!/usr/bin/env python
# coding: utf-8

# # Bank of Canada - Data and analytics specialist - written assessment
# 

# 
# ## Question A
# ### i) Download the Bitcoin price/volume/market cap data from this website: https://coinmarketcap.com/currencies/bitcoin/historical-data/ and reproduce the chart of Bitcoin to USD as shown in this link: https://coinmarketcap.com/currencies/bitcoin/. Be sure to include all three panels (Price, Market Cap, and Trading View); interactivity (hover, date expansion) is optional (extra 1 bonus point).
# 

# In[1]:


pip install mpl_finance


# In[2]:


import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.dates as mpl_dates
from mpl_finance import candlestick_ohlc


# In[66]:


df_bitcoin = pd.read_csv(os.path.join(sys.path[0], 'bitcoindata.csv'))


# In[67]:


df_bitcoin.head()


# In[57]:


# price chart
df_bitcoin.plot(kind='line', x='Date', y='Close', title='Price')


# In[58]:


# market cap chart
df_bitcoin.plot(kind='line', x='Date', y='Market Cap', title='Market Cap')


# In[59]:


# candlestick chart 
ohlc = df_bitcoin.loc[:, ['Date', 'Open', 'High', 'Low', 'Close']]
ohlc['Date'] = pd.to_datetime(ohlc['Date'])
ohlc['Date'] = ohlc['Date'].apply(mpl_dates.date2num)
ohlc = ohlc.astype(float)

# Creating Subplots
fig, ax = plt.subplots()

candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='green', colordown='red', alpha=0.8)

# Setting labels & titles
ax.set_xlabel('Date')
ax.set_ylabel('Price')
fig.suptitle('Trading View')

# Formatting Date
date_format = mpl_dates.DateFormatter('%d-%m-%Y')
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()

fig.tight_layout()

plt.show()


# ### v)	Using the same dataset and at least one additional data source of your choice, build any two visualizations using your choice of variables and complement them with notes and descriptions to explain why you have chosen these visualizations.

# #### Explore correlation between Bitcoin and S&P500 index. Per sources this implies that Bitcoin has become neither 'digital gold' nor a 'safe-haven asset' in times of crisis.
# 
# #### Limitation with this analysis is only able to retrieve most recent month worth of data for SP500 whereas have more historical for Bitcoin
# 
# 
# 

# In[79]:


df_bitcoinapr = pd.read_csv(os.path.join(sys.path[0], 'bitcoindataapril.csv'))
df_sp = pd.read_csv(os.path.join(sys.path[0], 'sp500.csv'))


# In[80]:


df_sp.head()


# In[84]:


#plt.plot(df_bitcoin.Date, df.bitcoin.Close)
f = plt.figure()
f.set_figwidth(30)
f.set_figheight(10)
plt.plot(df_bitcoinapr.Date, df_bitcoinapr.SPClose)
plt.plot(df_bitcoinapr.Date, df_bitcoinapr.Close)
plt.show()


# #### timeseries analysis with facebook prophet to predict next 60 days price bitcoin

# In[94]:


from fbprophet import Prophet


# In[95]:


datap = df_bitcoin.reset_index()


# In[96]:


# Select only the important features i.e. the date and price
datap = datap[["Date","Close"]] # select Date and Price
# Rename the features: These names are NEEDED for the model fitting
datap = datap.rename(columns = {"Date":"ds","Close":"y"}) #renaming the columns of the dataset
datap.head(5)


# In[97]:


m = Prophet(daily_seasonality = True) # the Prophet class (model)
m.fit(datap) # fit the model using all data


# In[105]:


future = m.make_future_dataframe(periods=60) #we need to specify the number of days in future
prediction = m.predict(future)
m.plot(prediction)
plt.title("Prediction of Bitcoin using the Prophet")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()


# ## Question B
# 

# ### i)	Clean the dataset found in the csv file, including interpolating missing datapoints. Please include a short explanation (maximum 50 words) of how you cleaned the data set.

# In[8]:


df_unrate = pd.read_csv(os.path.join(sys.path[0], 'unrate.csv'))


# In[9]:


df_unrate.head()


# In[10]:


# plot 3 individual years to get feel for the linear


# In[11]:


f = plt.figure()
f.set_figwidth(40)
f.set_figheight(10)  
  
plt.plot(df_unrate.iloc[:12].DATE, df_unrate.iloc[:12].UNRATE)
plt.show()


# In[12]:


f = plt.figure()
f.set_figwidth(40)
f.set_figheight(10)  
  
plt.plot(df_unrate.iloc[24:36].DATE, df_unrate.iloc[24:36].UNRATE)
plt.show()


# In[13]:


f = plt.figure()
f.set_figwidth(40)
f.set_figheight(10)  
  
plt.plot(df_unrate.iloc[36:48].DATE, df_unrate.iloc[36:48].UNRATE)
plt.show()


# In[14]:


f = plt.figure()
f.set_figwidth(40)
f.set_figheight(10)  
  
plt.plot(df_unrate.iloc[48:64].DATE, df_unrate.iloc[48:64].UNRATE)
plt.show()


# In[15]:


#histogram to get idea of distrbution
plt.hist(df_unrate.UNRATE, bins=12)
plt.show()


# In[16]:


#scatterplot to identify outliers
df_unrate.plot(kind='scatter', x='DATE', y='UNRATE', title='scatter')


# In[17]:


#load imputed values for unrate
df_unrate_rev = pd.read_csv(os.path.join(sys.path[0], 'unrate_rev.csv'))


# ### ii)	Create one table of the following summary statistics for the unemployment rate of the US: Mean, Standard Deviation, 25th Percentile, 50th Percentile, 75th Percentile, and Max. 

# In[18]:


import numpy as np  
np.percentile(df_unrate_rev.UNRATE, q=[0, 25, 50, 75, 100])  


# In[19]:


df_unrate_rev.describe()

