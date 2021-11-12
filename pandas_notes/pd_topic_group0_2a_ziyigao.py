#!/usr/bin/env python
# coding: utf-8

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

