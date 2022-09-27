#!/usr/bin/env python
# coding: utf-8

# # Statistics - Statistical Inference
# ## Student's T Distribution
# 
# Using data analysis and statistics to make conclusions about a population is called statistical inference.
# 
# The main types of statistical inference are:
# 
# + Estimation
# + Hypothesis testing
# 
# ### Hypothesis Testing
# Hypothesis testing is a method to check if a claim about a population is true. More precisely, it checks how likely it is that a hypothesis is true is based on the sample data.
# 
# There are different types of hypothesis testing.
# 
# The steps of the test depends on:
# 
# Type of data (categorical or numerical)
# If you are looking at:
# + A single group
# + Comparing one group to another
# + Comparing the same group before and after a change
# Some examples of claims or questions that can be checked with hypothesis testing:
# 
# >90% of Australians are left handed
# >Is the average weight of dogs more than 40kg?
# >Do doctors make more money than lawyers?
# 
# 
# 
# ### Hypothesis Testing a Proportion
# The following steps are used for a hypothesis test:
# 
# + Check the conditions
# + Define the claims
# + Decide the significance level
# + Calculate the test statistic
# + Conclusion
# 

# ### How to Generate a t Distribution
# The result is an array of 10 values that follow a t distribution with 6 degrees of freedom.
# 
# You can use the t.rvs(df, size) function to generate random values from a t distribution with a specific degrees of freedom and sample size:

# In[1]:


from scipy.stats import t

#generate random values from t distribution with df=6 and sample size=10
x = t.rvs(df=6, size=10)
x


# Example 1: Find One-Tailed P-Value
# 
# The one-tailed p-value that corresponds to a t test statistic of -1.5 with 10 degrees of freedom is 0.0822.
# 
# Suppose we perform a one-tailed hypothesis test and end up with a t test statistic of -1.5 and degrees of freedom = 10.
# 
# We can use the following syntax to calculate the p-value that corresponds to this test statistic:

# In[2]:


from scipy.stats import t

#calculate p-value
t.cdf(x=-1.5, df=10)


# ### Example 2: Find Two-Tailed P-Value
# 
# Suppose we perform a two-tailed hypothesis test and end up with a t test statistic of 2.14 and degrees of freedom = 20.
# 
# The two-tailed p-value that corresponds to a t test statistic of 2.14 with 20 degrees of freedom is 0.0448.
# 
# We can use the following syntax to calculate the p-value that corresponds to this test statistic:

# In[3]:


from scipy.stats import t

#calculate p-value
(1 - t.cdf(x=2.14, df=20)) * 2

0.04486555082549959


# ### How to Plot a t Distribution
# 
# You can use the following syntax to plot a t distribution with a specific degrees of freedom:

# In[4]:


from scipy.stats import t
import matplotlib.pyplot as plt

#generate t distribution with sample size 10000
x = t.rvs(df=12, size=10000)

#create plot of t distribution
plt.hist(x, density=True, edgecolor='black', bins=20, color='pink')


# In[4]:




