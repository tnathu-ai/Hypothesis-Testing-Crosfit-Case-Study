#!/usr/bin/env python
# coding: utf-8

# # Linear Regression
# 
# # Find n, Σy, Σx, Σxy, Σx^2 , Σy^2
# 
# + n: the number of data points
# + Σy: the sum of the y values
# + Σx: the sum of the x values
# + Σxy: the sum of the products of the x and y values
# + Σx^2: the sum of the squares of the x values
# + Σy^2: the sum of the squares of the y values

# In[35]:


# check the version of the python
get_ipython().system(' python --version')


# In[19]:


# Python program to demonstrate creating
# pandas Datadaframe from lists using zip.

import pandas as pd

# List1
x = [2, 4, 6, 8, 10, 12]

# List2
y = [1.8, 1.5, 1.4, 1.1, 1.1, 0.9]

# List3
# create xy by looping through x and y and multiplying them using list comprehension
xy = [x * y for x, y in zip(x, y)]

# List4
# create x2 by looping through x and squaring them using list comprehension
x2 = [x * x for x in x]

# List5
# create y2 by looping through y and squaring them using list comprehension
y2 = [y * y for y in y]

# List6
# sum of Σx by looping through x and adding them using list comprehension
sum_x = sum(x)

# List7
# sum of Σy by looping through y and adding them using list comprehension
sum_y = sum(y)

# List8
# sum of Σxy by looping through xy and adding them using list comprehension
sum_xy = sum(xy)

# List9
# sum of Σx^2 by looping through x2 and adding them using list comprehension
sum_x2 = sum(x2)

# List10
# sum of Σy^2 by looping through y2 and adding them using list comprehension
sum_y2 = sum(y2)


# get the list of tuples from two lists.
# and merge them by using zip().
list_of_tuples = list(zip(x, y, xy, x2, y2))

# Converting lists of tuples into
# pandas Dataframe.
df = pd.DataFrame(list_of_tuples,
                  columns=['x', 'y', 'xy', 'x²', 'y²'])

n = len(x)
print(f'The number of data n = {n}\nΣx = {sum_x}, Σy = {sum_y}, Σxy = {sum_xy}, Σx² = {sum_x2}, Σy² = {sum_y2}\n')
# Print data.
df


# # Find Sxy, Sxx, Syy
# 
# + Sxy: the sample covariance of x and y
# + Sxx: the sample variance of x
# + Syy: the sample variance of y
# 

# In[21]:


# Find Sxy, Sxx, and Syy
Sxy = sum_xy - (sum_x * sum_y) / n
Sxx = sum_x2 - (sum_x * sum_x) / n
Syy = sum_y2 - (sum_y * sum_y) / n
print(f'Sxy = {Sxy}, Sxx = {Sxx}, and Syy = {Syy}\n')


# # General regression equation
# y = a + bx OR y = b0 + b1x1 + b2x2 + b3x3 + ... + bnxn
# 
# 𝑦̂i=𝛽̂0+𝛽̂1𝑥i+εi

# In[24]:


# Find b1 and b0
b1 = Sxy / Sxx
b0 = (sum_y / n) - (b1 * (sum_x / n))
print(f'b1 = {b1}, b0 = {b0}\nThe regression equation is y = {b0} + ({b1}x)')


# # Find SSTO, SSR, SSE, R2, and R
# 
# + SSTO: Total Sum of Squares
# + SSR: Regression Sum of Squares
# + SSE: Error Sum of Squares
# + R2: Coefficient of Determination
# + R: Correlation Coefficient
# 

# In[32]:


from cmath import sqrt

# determine SSTO, SSR, SSE, R^2, and R
# The sum squares of the total:
# SSTO = SSR + SSE
SSTO = Syy

# The sum squares of the regression:
# SSR = b1 * Sxy
# SSR = (b1^2) * Sxx
SSR = (Sxy*Sxy) / Sxx

# The sum squares of the error:
SSE = Syy - SSR
# SSE = ((Sxx * SSyy) - (Sxy*Sxy))/Sxx

# The coefficient of determination:
R2 = SSR / SSTO
# R2 = 1 - (SSE / SSTO)
print(f'{R2*100}% of the variation in y is explained by the variation in x')

# The correlation coefficient:
R = sqrt(R2)

print(f'SSTO = {SSTO}\n, SSR = {SSR}\n, SSE = {SSE}\n, R^2 = {R2}\n, and R = {R}\n')


# # Analysis of Variance (ANOVA) for Regression
# 
# | **Source**     | **DF** | **Adj SS** | **Adj MS**        | **F-Value** | **Note**                               |
# |:--------------:|:------:|:----------:|:-----------------:|:-----------:|:--------------------------------------:|
# | **Regression** | k      | SSR        | MSR = SSR/k       | F = MSR/MSE | k: number of independent variables (x) |
# | **Error**      | n-1-k  | SSE        | MSE = SSE/(n-k-1) |             | n: sample size                         |
# | **Total**      | (n-1)  | SST        |                   |             |                                        |
# 
# # Assumption:
# + Each of the error between the actual data and predicted one is normally distributed
# + The independent variables (x) are not random
# 
# **Note here used the t-table instead of the z-table because we dont know the standard deviation for the population despite the large sample.**
# 
# # Hypothesis test:
# + H₀: β1 = 0
# + H₁: β1 ≠ 0  -> there's evidence of relationship between x and y
# 
# # Critical value (t-table)
# t(α/2;n-2)
# 
# `scipy.stats.t.ppf(q, df)`
# 
# + q: The critical level to use.
# + df: The degrees of freedom.

# In[45]:


# Find t critical

# Import Library
import scipy.stats

# n = dfT + 1
critical_level = 0.1
degrees_freedom = 11  # degrees_freedom = n-2

# Right-tailed test to find the T critical value
right_tailed = scipy.stats.t.ppf(q=1-critical_level,df=degrees_freedom)
# Left-tailed test to find the T critical value
left_tailed = scipy.stats.t.ppf(q=critical_level,df=degrees_freedom)
# 2 tailed to find the T critical value
two_tailed = scipy.stats.t.ppf(q=1-critical_level/2,df=degrees_freedom)

print(f'Right-tailed test: {right_tailed}\nLeft-tailed test: {left_tailed}\n2 tailed test: {two_tailed}')


# # Test statistic

# In[ ]:
# Minimize the error
# Find the test statistic
# t = b1 / SEb1
# SEb1 = sqrt(MSE / Sxx)
# MSE = SSE / (n-k-1)
# k = 1
# n = 12





# In[43]:


import os

# The .py format of the jupyter notebook
for fname in os.listdir():
    if fname.endswith('ipynb'):
        os.system(f'jupyter nbconvert {fname} --to python')


# # References
# + https://www.statisticshowto.com/probability-and-statistics/regression-analysis/anova-for-regression/
# + https://www.geeksforgeeks.org/how-to-find-the-t-critical-value-in-python/

# In[ ]:




