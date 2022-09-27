#!/usr/bin/env python
# coding: utf-8

# <a id="1"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Normality Test</strong></h1>
# 
# 

# # Importing Necessary Libraries and datasets

# In[33]:


# Install a conda package in the current Jupyter kernel
get_ipython().system('{sys.executable} -m pip install statsmodels')

# work with df in tabular representation
import pandas as pd
# round the df in the correlation matrix
import numpy as np
import os
from scipy.stats import t
from scipy import stats


# Modules for df visualization
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import shapiro
from numpy import mean
from numpy import std

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


# In[34]:


# check the version of the packages
print("Numpy version: ", np.__version__)
print("Pandas version: ",pd.__version__)
get_ipython().system(' python --version')


# In[35]:


# set the general path of the external df
external_df_path = os.path.join(os.path.pardir,'data','interim')

# set the path for specific dfset from external dfset
df = os.path.join(external_df_path, 'cleaned_data.csv')


# In[36]:


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

# In[37]:


# see the static of all numerical column
df.describe().T


# In[38]:


# Number of Unique Athlete
print(df.athlete_id.nunique() == df.shape[0])
# Summary Stats: weight
df.describe([0.01, 0.05, 0.10, 0.20, 0.80, 0.90, 0.95, 0.99])[["weight"]].T


# # The Central Limit Theorem (CLT)
# 
# The problem of small samples is a common one, and was first investigated by William Gosset who is better known by his pseudonym Student. Gosset worked for the Guinness brewing company and was trying to compare strains of barley. In his experiments different fields of barley would each yield one data point. Clearly you don’t want to cultivate 40 fields if ten would do. The tests we will describe in this section assume that the population being sampled is
# Normally distributed. If this is assumed then the distribution of sample means, ̄X, will also be normally distributed with the same mean and variance σ2/n. But in practice we do not know σ, we need to estimate it from the sample
# data, that is we use s to estimate σ. This works for large enough n, but s is not a good approximation to the unknown σ for small n. Now to do hypothesis tests, and other things concerning the unknown mean of the population, we need to calculate
# 
# ![The Central Limit Theorem (CLT) formula](../media/images/The_Central_Limit_Theorem.png)
# ![The Central Limit Theorem (CLT) definition](../media/images/CLT_definition.png)
# 
# 
# Data is whether to use parametric or nonparametric statistical methods.
# RULE OF THUMB
# If n > 30, the Central Limit Theorem can be used.
# Of course, exceptions exist, but this rule applies to most distributions of real data.

# In[39]:


# histogram plot
from numpy.random import seed
from numpy.random import randn
from matplotlib import pyplot
# seed the random number generator
seed(1)
# generate univariate observations
data = 5 * randn(100) + 50
# histogram plot
pyplot.hist(data, color="pink")
pyplot.show()


# In[40]:


plt.figure(figsize = (20,10))
sns.distplot(df.weight,color = 'pink')
plt.savefig('distplot.png')


# In[41]:


#  Groups & Target Summary Stats
df.groupby("gender").weight.agg(["count", "median", "mean", "std", "max"])


# <a id="1"></a> <br>
# # Histogram
# * How many times each value appears in dfset. This description is called the distribution of variable
# * Most common way to represent distribution of varible is histogram that is graph which shows frequency of each value.
# * Frequency = number of times each value appears

# In[42]:


import plotly.express as px

df1 = df[['weight','gender']]
fig = px.histogram(df1, x="gender",y="weight",  histfunc='avg')
fig.show()


# In[43]:


female = plt.hist(df[df["gender"] == "female"].weight,bins=30,label = "female", color="pink")
male = plt.hist(df[df["gender"] == "male"].weight,bins=30,label = "male", color="lightblue")
plt.legend()
plt.xlabel(" Mean Values")
plt.ylabel("Frequency")
plt.title("Histogram of  Mean for male and female Weight")
plt.savefig('female_male_weight_hist.png')
plt.show()
frequent_weight_mean = female[0].max()
index_frequent_weight_mean = list(female[0]).index(frequent_weight_mean)
most_frequent_weight_mean = female[1][index_frequent_weight_mean]
print("Most frequent female weight mean is: ",most_frequent_weight_mean)


# # How to Plot a t Distribution with a specific degrees of freedom:

# In[44]:


#generate t distribution with sample size 10
x = t.rvs(df=9, size=10)

#create plot of t distribution
plt.hist(x, density=True, edgecolor='black', bins=20, color='pink')


# ---------
# # Randomly select a 10 number of rows from a dataframe
# 

# In[45]:


df_sample = df.sample(n=10, random_state=42)
df_sample


# In[46]:


# convert specified column in the dataframe into series
population_weight = df['weight'].squeeze()
print('mean=%.3f stdv=%.3f' % (mean(population_weight), std(population_weight)))
sample_weight = df_sample['weight'].squeeze()
print('mean=%.3f stdv=%.3f' % (mean(sample_weight), std(sample_weight)))


# # 1. Normality Tests
# 
# I need to decide whether to use parametric or nonparametric statistical methods.
# 
# This section lists statistical tests that you can use to check if your data has a Gaussian distribution.
# 
# Shapiro-Wilk Test
# 
# Tests whether a data sample has a Gaussian distribution.
# 
# Assumptions
# 
# Observations in each sample are independent and identically distributed (iid).
# Interpretation
# 
# H0: the sample has a Gaussian distribution.
# H1: the sample does not have a Gaussian distribution.

# # Shapiro-Wilk Test
# 
# The Shapiro-Wilk test evaluates a data sample and quantifies how likely it is that the data was drawn from a Gaussian distribution, named for Samuel Shapiro and Martin Wilk.
# 
# In practice, the Shapiro-Wilk test is believed to be a reliable test of normality, although there is some suggestion that the test may be suitable for smaller samples of data, e.g. thousands of observations or fewer.
# 
# The shapiro() SciPy function will calculate the Shapiro-Wilk on a given dataset. The function returns both the W-statistic calculated by the test and the p-value.
# 
# The complete example of performing the Shapiro-Wilk test on the dataset is listed below.

# In[47]:


# Shapiro-Wilk Test
# normality test
stat, p = shapiro(df.weight)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


# In[48]:


shapiro(df.weight)


# In[49]:


def check_normality_shapiro(data):
    test_stat_normality, p_value_normality=stats.shapiro(data)
    print("p value:%.3f" % p_value_normality)
    stat, p = shapiro(df.weight)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    if p_value_normality <0.05:
        print("Reject null hypothesis: The data is not normally distributed")
    else:
        print("Fail to reject null hypothesis: The data is normally distributed")


# In[50]:


check_normality_shapiro(population_weight)


# In[51]:


check_normality_shapiro(sample_weight)


# In[52]:


def check_variance_homogeneity(group1, group2):
    test_stat_var, p_value_var= stats.levene(group1,group2)
    print("p value:%.3f" % p_value_var)
    if p_value_var <0.05:
        print("Reject null hypothesis: The variances of the samples are different.")
    else:
        print("Fail to reject null hypothesis: The variances of the samples are same.")


# In[53]:


check_variance_homogeneity(population_weight, sample_weight)


# # T-Sample Test

# In[54]:


ttest,p_value = stats.ttest_ind(np.array(population_weight).astype(int), np.array(sample_weight).astype(int))
print("p value:%.3f" % p_value)
if p_value <0.05:
    print("Reject null hypothesis")
else:
    print("Fail to reject null hypothesis")


# Conclusion
# 
# For a variety of reasons, my data could not be normally distributed. Each test takes a somewhat different approach to the research question of whether a sample was taken from a Gaussian distribution.
# 
# Your data are not normal if even one normality test fails, just like that.
# 
# You can either look into the cause of your data's non-normal behavior and possibly implement data preparation procedures to restore the data's normal behavior.
# 
# Alternately, you may start investigating the usage of nonparametric statistical techniques in place of parametric ones.

# # References
# + Docs.scipy.org. 2022. Statistical functions (scipy.stats) — SciPy v1.9.0 Manual. [online] Available at: <https://docs.scipy.org/doc/scipy/reference/stats.html> [Accessed 26 August 2022].

# In[ ]:




