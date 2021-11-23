#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np


# 
# Ziyi Gao
# 
# ziyigao@umich.edu
# 
# ## Multi-indexing
# 
# - Aiming at sophisticated data analysis and manipulation, especially for working with higher dimensional data
# - Enabling one to store and manipulate data with an arbitrary number of dimensions in lower dimensional data structures
# 
# ## Creating a multi-indexing dataframe and Reconstructing
# 
# - It can be created from:
#     - a list of arrays (using MultiIndex.from_arrays())
#     - an array of tuples (using MultiIndex.from_tuples())
#     - a crossed set of iterables (using MultiIndex.from_product())
#     - a DataFrame (using MultiIndex.from_frame())
# - The method get_level_values() will return a vector of the labels for each location at a particular level
# 
# ## Basic Indexing
# 
# - Advantages of hierarchical indexing
#     - hierarchical indexing can select data by a “partial” label identifying a subgroup in the data
# - Defined Levels
#     - keeps all the defined levels of an index, even if they are not actually used
#     
# ## Data Alignment and Using Reindex
# 
# - Operations between differently-indexed objects having MultiIndex on the axes will work as you expect; data alignment will work the same as an Index of tuples
# - The reindex() method of Series/DataFrames can be called with another MultiIndex, or even a list or array of tuples:
# 
# ## Some Advanced Indexing
# 
# Syntactically integrating MultiIndex in advanced indexing with .loc is a bit challenging
# 
# - In general, MultiIndex keys take the form of tuples
#     

# In[ ]:


import pandas as pd
import numpy as np
### Q0 code example

#created from arrays or tuples

arrays = [["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
          ["one", "two", "one", "two", "one", "two", "one", "two"]]
tuples = list(zip(*arrays)) # if from arrays, this step is dropped
index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"]) 
# if from arrays, use pd.MultiIndex.from_arrays()

df1 = pd.Series(np.random.randn(8), index=index)

#created from product

iterables = [["bar", "baz", "foo", "qux"], ["one", "two"]]
df2 = pd.MultiIndex.from_product(iterables, names=["first", "second"])

#created directly from dataframe
df3 = pd.DataFrame([["bar", "one"], ["bar", "two"], ["foo", "one"], ["foo", "two"]],
                  columns=["first", "second"])
pd.MultiIndex.from_frame(df)

#Basic Operation and Reindex

df1 + df1[:2]
df1 + df1[::2]

df1.reindex(index[:3])
df1.reindex([("foo", "two"), ("bar", "one"), ("qux", "one"), ("baz", "one")])

#Advanced Indexing 
df1 = df1.T
df1.loc[("bar", "two")]


# ## Topics in Pandas
# **Stats 507, Fall 2021** 
#   
# 
# ## Contents
# Add a bullet for each topic and link to the level 2 title header using 
# the exact title with spaces replaced by a dash. 
# 
# + [Pandas Query](#Pandas-Query) 
# + [Time Series](#Time-Series) 
# + [Window Functions](#Window-Functions) 
# 
# ## Pandas Query ##
# 
# ### pd. query ##
# 
# ###### Name: Anandkumar Patel
# ###### Email: patelana@umich.edu
# ###### Unique ID: patelana
# 
# ### Arguments and Output
# 
# **Arguments** 
# 
# * expression (expr) 
# * inplace (default = False) 
#     * Do you want to operate directly on the dataframe or create new one
# * kwargs (keyword arguments)
# 
# **Returns** 
# * Dataframe from provided query
# 
# ## Why
# 
# * Similar to an SQL query 
# * Can help you filter data by querying
# * Returns a subset of the DataFrame
# * loc and iloc can be used to query either rows or columns
# 
# ## Query Syntax
# 
# * yourdataframe.query(expression, inplace = True/False
# 
# ## Code Example

# In[2]:

# In[ ]:


import pandas as pd
df = pd.DataFrame({'A': range(1, 6),
                   'B': range(10, 0, -2),
                   'C C': range(10, 5, -1)})
print(df)

print('Below is the results of the query')

print(df.query('A > B'))


# ## Time Series
# **Name: Lu Qin**
# UM email: qinlu@umich.edu
# 
# ### Overview
#  - Data times
#  - Time Frequency
#  - Time zone
# 
# ### Import

# In[ ]:


import datetime
import pandas as pd
import numpy as np


# ### Datetime
#  - Parsing time series information from various sources and formats

# In[ ]:


dti = pd.to_datetime(
    ["20/10/2021", 
     np.datetime64("2021-10-20"), 
     datetime.datetime(2021, 10, 20)]
)

dti


# ### Time frequency
# - Generate sequences of fixed-frequency dates and time spans
# - Resampling or converting a time series to a particular frequency

# #### Generate

# In[ ]:


dti = pd.date_range("2021-10-20", periods=2, freq="H")

dti


# #### convert

# In[ ]:


idx = pd.date_range("2021-10-20", periods=3, freq="H")
ts = pd.Series(range(len(idx)), index=idx)

ts


# #### resample

# In[ ]:


ts.resample("2H").mean()


# ### Timezone
#  - Manipulating and converting date times with timezone information
#  - `tz_localize()`
#  - `tz_convert()`

# In[ ]:


dti = dti.tz_localize("UTC")
dti

dti.tz_convert("US/Pacific")


# ## Window Functions ##
# **Name: Stephen Toner** \
# UM email: srtoner@umich.edu

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web


# Of the many funcitons in Pandas, one which is particularly useful for time
# series analysis is the window function. It lets us apply some aggregation 
# function over a specified lookback period on a rolling basis throughout the
# time series. This is particularly useful for financial analsyis of equity
# returns, so we will compute some financial metrics for Amazon stock using
# this techinique.

# Our first step is to import our data for Amazon ("AMZN") 
# over a healthy time horizon:

# In[ ]:


amzn_data = web.DataReader("AMZN", 
                           data_source = 'yahoo', 
                           start = "2016-10-01", 
                           end = "2021-10-01")

amzn_data.head()


# While the column labels are largely self-explanatory, two important notes
# should be made:
# * The adjusted close represents the closing price after all is said and done
# after the trading session ends; this may represent changes due to accounts 
# being settled / netted against each other, or from adjustments to financial
# reporting statements.
# * One reason for our choice in AMZN stock rather than others is that AMZN
# has not had a stock split in the last 20 years; for this reason we do not
# need to concern ourselves with adjusting for the issuance of new shares like
# we would for TSLA, AAPL, or other companies with large
# market capitalization.

# Getting back to Pandas, we have three main functions that allow us to
# perform Window operations:
# * `df.shift()`: Not technically a window operation, but helpful for
# computing calculations with offsets in time series
# * `rolling`: For a given fixed lookback period, tells us the 
# aggregation metric (mean, avg, std dev)
# * `expanding`: Similar to `rolling`, but the lookback period is not fixed. 
# Helpful when we want to have a variable lookback period such as "month to 
# date" returns

# Two metrics that are often of interest to investors are the returns of an
# asset and the volume of shares traded. Returns are either calculated on
# a simple basis:
# $$ R_s = P_1/P_0 -1$$
# or a log basis:
# $$ R_l = \log (P_1 / P_2) $$
# Simple returns are more useful when aggregating returns across multiple 
# assets, while Log returns are more flexible when looking at returns across 
# time. As we are just looking at AMZN, we will calculate the log returns
# using the `shift` function:

# In[ ]:


amzn_data["l_returns"] = np.log(amzn_data["Adj Close"]/
                                amzn_data["Adj Close"].shift(1))


plt.title("Log Returns of AMZN")
plt.plot(amzn_data['l_returns'])


# For the latter, we see that the
# volume of AMZN stock traded is quite noisy:

# In[ ]:


plt.title("Daily Trading Volume of AMZN")   
plt.plot(amzn_data['Volume'])


# If we want to get a better picture of the trends, we can always take a
# moving average of the last 5 days (last full set of trading days):

# In[ ]:


amzn_data["vol_5dma"] = amzn_data["Volume"].rolling(window = 5).mean()
plt.title("Daily Trading Volume of AMZN")   
plt.plot(amzn_data['vol_5dma'])


# When we apply this to a price metric, we can identify some technical patterns
# such as when the 15 or 50 day moving average crosses the 100 or 200 day
# moving average (known as the golden cross, by those who believe in it).

# In[ ]:


amzn_data["ma_15"] = amzn_data["Adj Close"].rolling(window = 15).mean()
amzn_data["ma_100"] = amzn_data["Adj Close"].rolling(window = 100).mean()

fig1 = plt.figure()
plt.plot(amzn_data["ma_15"])
plt.plot(amzn_data["ma_100"])
plt.title("15 Day MA vs. 100 Day MA")

# We can then use the `shift()` method to identify which dates have 
# golden crosses

gc_days = (amzn_data.eval("ma_15 > ma_100") & 
               amzn_data.shift(1).eval("ma_15 <= ma_100"))

gc_prices = amzn_data["ma_15"][gc_days]


fig2 = plt.figure()
plt.plot(amzn_data["Adj Close"], color = "black")
plt.scatter( x= gc_prices.index, 
                y = gc_prices[:],
                marker = "+", 
                color = "gold" 
                )

plt.title("Golden Crosses & Adj Close")


# The last feature that Pandas offers is a the `expanding` window function, 
# which calculates a metric over a time frame that grows with each additional 
# period. This is particularly useful for backtesting financial metrics
# as indicators of changes in equity prices: because one must be careful not
# to apply information from the future when performing backtesting, the 
# `expanding` functionality helps ensure we only use information up until the 
# given point in time. Below, we use the expanding function to plot cumulative
# return of AMZN over the time horizon.

# In[ ]:


def calc_total_return(x):
    """    
    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.log(x[-1] / x[0]) 


amzn_data["Total Returns"] = (amzn_data["Adj Close"]
                              .expanding()
                              .apply(calc_total_return))

fig3 = plt.figure()
ax5 = fig3.add_subplot(111)
ax5 = plt.plot(amzn_data["Total Returns"])
plt.title("Cumulative Log Returns for AMZN")


# ## Contents
# Add a bullet for each topic and link to the level 2 title header using 
# the exact title with spaces replaced by a dash. 
# 
# + [Topic Title](#Topic-Title)
# + [Topic 2 Title](#Topic-2-Title)
# 
# ## Topic Title
# Include a title slide with a short title for your content.
# Write your name in *bold* on your title slide. 
# 
# 
# 
# ## Contents
# + [Processing Time Data](#Processing-Time-Data)
# + [Topic 2 Title](#Topic-2-Title)
# 
# * ###  Processing Time Data
# 
# **Yurui Chang**
# 
# #### Pandas.to_timedelta
# 
# - To convert a recognized timedelta format / value into a Timedelta type
# - the unit of the arg
#   * 'W'
#   * 'D'/'days'/'day'
#   * ‘hours’ / ‘hour’ / ‘hr’ / ‘h’
#   * ‘m’ / ‘minute’ / ‘min’ / ‘minutes’ / ‘T’
#   * ‘S’ / ‘seconds’ / ‘sec’ / ‘second’
#   * ‘ms’ / ‘milliseconds’ / ‘millisecond’ / ‘milli’ / ‘millis’ / ‘L’
#   * ‘us’ / ‘microseconds’ / ‘microsecond’ / ‘micro’ / ‘micros’ / ‘U’
#   * ‘ns’ / ‘nanoseconds’ / ‘nano’ / ‘nanos’ / ‘nanosecond’ / ‘N’
# 
# * Parsing a single string to a Timedelta
# * Parsing a list or array of strings
# * Converting numbers by specifying the unit keyword argument

# In[3]:


time1 = pd.to_timedelta('1 days 06:05:01.00003')
time2 = pd.to_timedelta('15.5s')
print([time1, time2])
pd.to_timedelta(['1 days 06:05:01.00003', '15.5s', 'nan'])

pd.to_timedelta(np.arange(5), unit='d')


# #### pandas.to_datetime
# 
# * To convert argument to datetime
# * Returns: datetime, return type dependending on input
#   * list-like: DatetimeIndex
#   * Series: Series of datetime64 dtype
#   * scalar: Timestamp
# * Assembling a datetime from multiple columns of a DataFrame
# * Converting Pandas Series to datetime w/ custom format
# * Converting Unix integer (days) to datetime
# * Convert integer (seconds) to datetime

# In[ ]:


s = pd.Series(['date is 01199002',
           'date is 02199015',
           'date is 03199020',
           'date is 09199204'])
pd.to_datetime(s, format="date is %m%Y%d")

time1 = pd.to_datetime(14554, unit='D', origin='unix')
print(time1)
time2 = pd.to_datetime(1600355888, unit='s', origin='unix')
print(time2)


# # Title: Pandas Time Series Analysis
# ## Name: Kenan Alkiek (kalkiek)

# In[ ]:


from matplotlib import pyplot as plt

# Read in the air quality dataset
air_quality = pd.read_csv(
    'https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/air_quality_no2_long.csv')
air_quality["datetime"] = pd.to_datetime(air_quality["date.utc"])

# One common method of dealing with time series data is to set the index equal to the data
air_quality = air_quality.set_index('datetime')
air_quality.head()

# Plot the NO2 Over time for Paris france
paris_air_quality = air_quality[(air_quality['city'] == 'Paris') & (air_quality['country'] == 'FR')]

paris_air_quality.plot()
plt.ylabel("$NO_2 (µg/m^3)$")

# Plot average NO2 by hour of the day
fig, axs = plt.subplots(figsize=(12, 4))
air_quality.groupby("date.utc")["value"].mean().plot(kind='bar', rot=0, ax=axs)
plt.xlabel("Hour of the day")
plt.ylabel("$NO_2 (µg/m^3)$")
plt.show()

# Limit the data between 2 dates
beg_of_june = paris_air_quality["2019-06-01":"2019-06-03"]
beg_of_june.plot()
plt.ylabel("$NO_2 (µg/m^3)$")

# Resample the Data With a Different Frequency (and Aggregration)
monthly_max = air_quality.resample("M").max()
print(monthly_max)

# Ignore weekends and certain times
rng = pd.date_range('20190501 09:00', '20190701 16:00', freq='30T')

# Grab only certain times
rng = rng.take(rng.indexer_between_time('09:30', '16:00'))

# Remove weekends
rng = rng[rng.weekday < 5]

rng.to_series()


# ## Pivot Table in pandas
# 
# 
# *Mingjia Chen* 
# mingjia@umich.edu
# 
# - A pivot table is a table format that allows data to be dynamically arranged and summarized in categories.
# - Pivot tables are flexible, allowing you to customize your analytical calculations and making it easy for users to understand the data.
# - Use the following example to illustrate how a pivot table works.

# In[ ]:


import numpy as np

df = pd.DataFrame({"A": [1, 2, 3, 4, 5],
                   "B": [0, 1, 0, 1, 0],
                   "C": [1, 2, 2, 3, 3],
                   "D": [2, 4, 5, 5, 6],
                   "E": [2, 2, 4, 4, 6]})
print(df)


# ## Index
# 
# - The simplest pivot table must have a data frame and an index.
# - In addition, you can also have multiple indexes.
# - Try to swap the order of the two indexes, the data results are the same.

# In[ ]:


tab1 = pd.pivot_table(df,index=["A"])
tab2 = pd.pivot_table(df,index=["A", "B"])
tab3 = pd.pivot_table(df,index=["B", "A"])
print(tab1)
print(tab2)
print(tab3)


# ## Values 
# - Change the values parameter can filter the data for the desired calculation.

# In[ ]:


pd.pivot_table(df,index=["B", "A"], values=["C", "D"])


# ## Aggfunc
# 
# - The aggfunc parameter sets the function that we perform when aggregating data.
# - When we do not set aggfunc, it defaults aggfunc='mean' to calculate the mean value.
#   - When we also want to get the sum of the data under indexes:

# In[ ]:


pd.pivot_table(df,index=["B", "A"], values=["C", "D"], aggfunc=[np.sum,np.mean])


# ## Columns
# 
# - columns like index can set the column hierarchy field, it is not a required parameter, as an optional way to split the data.
# 
# - fill_value fills empty values, margins=True for aggregation

# In[ ]:


pd.pivot_table(df,index=["B"],columns=["E"], values=["C", "D"],
               aggfunc=[np.sum], fill_value=0, margins=1)

