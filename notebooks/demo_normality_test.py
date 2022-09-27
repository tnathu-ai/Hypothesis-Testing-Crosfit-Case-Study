#!/usr/bin/env python
# coding: utf-8

# <a id="1"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Normality Test</strong></h1>
# 
# 

# # Importing Necessary Libraries and datasets

# In[1]:


# Install a conda package in the current Jupyter kernel
get_ipython().system('{sys.executable} -m pip install statsmodels')

# work with df in tabular representation
import pandas as pd
# round the df in the correlation matrix
import numpy as np
import os
from scipy.stats import t

# Modules for df visualization
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import shapiro
from numpy import mean
from numpy import std
from numpy.random import seed
from numpy.random import randn
from matplotlib import pyplot

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

plt.rcParams['figure.figsize'] = [6, 6]

# Ensure that our plots are shown and embedded within the Jupyter notebook itself. Without this command, sometimes plots may show up in pop-up windows
get_ipython().run_line_magic('matplotlib', 'inline')

# overwrite the style of all the matplotlib graphs
sns.set()

# ignore DeprecationWarning Error Messages
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# check the version of the packages
print("Numpy version: ", np.__version__)
print("Pandas version: ",pd.__version__)
get_ipython().system(' python --version')


# In[ ]:


# set the general path of the external df
external_df_path = os.path.join(os.path.pardir,'data','interim')

# set the path for specific dfset from external dfset
df = os.path.join(external_df_path, 'cleaned_data.csv')


# In[ ]:


# import dfset
df = pd.read_csv(df, delimiter=',', skipinitialspace = True)

# convert columns to the best possible dtypes, object->string
df = df.convert_dtypes()

# select numeric columns
df_numeric = df.select_dtypes(include=[np.number]).columns.to_list()

# select non-numeric columns
df_string = df.select_dtypes(include='string').columns.tolist()


print("Numeric columns: ", df_numeric, "\n")
print("String columns: ", df_string, "\n\n")

# print dfset info
print("The shape and df type of the ORGINAL df:", str(df.info()))

# print first 3 rows
df.head(3)


# # DESCRIPTIVE STATISTIC

# In[ ]:


# see the static of all numerical column


# In[ ]:


# Number of Unique Athlete

# Summary Stats: weight


# # The Central Limit Theorem (CLT)
# 
# 
# ![The Central Limit Theorem (CLT) formula](../media/images/The_Central_Limit_Theorem.png)

# In[ ]:


# histogram plot
# seed the random number generator


# In[ ]:


# displot graph


# In[ ]:


#  Groups & Target Summary Stats


# <a id="1"></a> <br>
# # Histogram
# * How many times each value appears in dfset. This description is called the distribution of variable
# * Most common way to represent distribution of varible is histogram that is graph which shows frequency of each value.
# * Frequency = number of times each value appears

# In[ ]:


# histogram plot using plotly


# In[ ]:


# hist group by gender


# # How to Plot a t Distribution with a specific degrees of freedom:

# In[ ]:


#generate t distribution with sample size 10

#create plot of t distribution


# ---------
# # Randomly select a 10 number of rows from a dataframe
# 

# In[ ]:


# sample 10 samples


# In[ ]:


# convert specified column in the dataframe into series
# get mean for population and sample


# # 1. Normality Tests
# 
# I need to decide whether to use parametric or nonparametric statistical methods.
# Assumptions
# 
# Observations in each sample are independent and identically distributed
# 
# H0: the sample has a Gaussian distribution.
# H1: the sample does not have a Gaussian distribution.

# # Shapiro-Wilk Test
# The function returns both the W-statistic calculated by the test and the p-value.

# In[ ]:


# Shapiro-Wilk Test
# normality test


# # Conclusion

# In[ ]:





# In[ ]:





# In[ ]:




