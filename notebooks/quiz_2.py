#!/usr/bin/env python
# coding: utf-8

# + Comparing Two Population Proportions
# # Nonparametric Statistical Significance Tests
# + Confidence Interval for the Difference of two Proportions
# + Hypothesis Test for the Difference of two Proportions
# + Better Method for Computing Standard Deviation when H0: p_1=p_2
# + Wilcoxon Signed-Rank Test
# + Wilcoxon Rank-Sum Test (Mann Whitney Test)
# + Chi-Square Distribution
# + Confidence Interval for Chi-Square
# + Hypothesis Test for a Single Variance
# + F Distribution
# + Hypothesis Test for Two Variances
# 

# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Comparing 2 Population Proportions</strong></h1>
# 
# # Central Limit Theorem
# ![Theorem 1 and 2 - CLT](../media/images/CLT_1_2.png)
# 
# `pf-pm` where
# + `pf` is the proportion of females with the rank higher
# + `pm` is the proportion of males of female
# 
# 1. Know the probability distribution for the random variable (p hat f) - (p hat m)
# 2. find mean & standard deviation
# ![Theorem 2 - Find mean and standard deviation](../media/images/CLT_2.png)

# <a id="1"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Confidence Interval for the Difference of two Proportions</strong></h1>
# 
# 
# **note:**
# + the populations need to be independent
# + p^1 & p^2 have to be approximately normally distributed
# + 3 samples need to be random
# 
# ### test for the above conditions
# ![CI for 2 Proportions](../media/images/CI_2_Proportions.png)

# <a id="1"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Hypothesis Test for the Difference of two Proportions</strong></h1>
# 
# ![Hypothesis Test for the Difference of 2 Proportions](../media/images/hypothesis_test_2_proportions.png)
# 

# <a id="1"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Better Method for Computing Standard Deviation when H0: p_1=p_2</strong></h1>

# In[38]:





# <a id="1"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Wilcoxon Signed-Rank Test</strong></h1>
# 
# **(paired data)**
# 
# equivalent of the paired Student T-test, but for ranked data instead of real valued data with a Gaussian distribution.
# 
# 
# + Fail to Reject H0: Sample distributions are equal.
# + Reject H0: Sample distributions are not equal.

# In[39]:


# Wilcoxon signed-rank test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import wilcoxon
# seed the random number generator
seed(1)
# generate two independent samples
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 51
# compare samples
stat, p = wilcoxon(data1, data2)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')


# <a id="1"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Wilcoxon Rank-Sum Test (Mann Whitney Test)</strong></h1>
# 
# The Mann-Whitney U test is a nonparametric statistical significance test for determining whether two independent samples were drawn from a population with the same distribution
# 
# + Fail to Reject H0: Sample distributions are equal.
# + Reject H0: Sample distributions are not equal.

# In[40]:


# Mann-Whitney U test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import mannwhitneyu
# seed the random number generator
seed(1)
# generate two independent samples
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 51
# compare samples
stat, p = mannwhitneyu(data1, data2)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')


# <a id="1"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Chi-Square Distribution (X² Distribution)</strong></h1>
# 
# [Chi-Squared Distribution Table](https://people.smp.uq.edu.au/YoniNazarathy/stat_models_B_course_spring_07/distributions/chisqtab.pdf)
# 
# ![chi_squared_distribution](../media/images/chi_squared_distribution.png)
# 
# + σ2: population variance
# + s2: sample variance
# + α: significance level
# + n: number of samples
# df: degrees of freedom.
# + χ2: chi-squared statistic. Depends on α and df

# ### To find the Chi-Square critical value, you need:
# 
# **NOTE:** the table always give us the probability of the right side of the graph. However, Minitab and Python chi-squared function give the probabilty on the left
# 
# + A significance level (common choices are 0.01, 0.05, and 0.10)
# + Degrees of freedom
# 
# `scipy.stats.chi2.ppf(q, df)`
# 
# where:
# + q: The significance level to use
# + df: The degrees of freedom
# 
# **find Chi-Square critical value for an upper tail with &alpha;=0.1 and &nu;=7**
# + Choose X²(0.1,7) in the table
# 

# In[41]:


# Find the Chi-Square Critical Value
import scipy.stats

#find Chi-Square critical value
scipy.stats.chi2.ppf(1-.1, df=7) #12.0170


# **find Chi-Square critical value for an lower tail where the area of the tail is 0.1 (&alpha;=0.9) and &nu;=7**
# + From the tables we see that the area reported is always the area to the right of the specified value.
# + So the boundary for the lower tail is found by looking up to X²(0.9,7) in the table

# In[42]:


# Find the Chi-Square Critical Value
import scipy.stats

#find Chi-Square critical value
scipy.stats.chi2.ppf(.1, df=7) # 2.83311


# **Find Chi-Square critical value for 2 tailed hypothesis tests and confidence intervals. Find the X²L and X²U values such that `P(X²L <= X² <= X²U)=0.95` where we assume that the areas in the upper and lower tails that have been ommited are equal**
# 
# + We need to split up &alpha; in 2 parts such as there is an &alpha;/2 in the upper and &alpha;/2 in the lower tail.
# + The critical X² value for the upper tail, where the area of the tail is 0.025 (&alpha;=0.025) and &nu;=24. So we look up X²(0.025,24) in the table
# + The critical X² value for the lower tail, where the area of the tail is 1-0.025 (&alpha;=0.975) and &nu;=24. So we look up X²(0.975,24) in the table

# In[53]:


# find Chi-Square critical value for 2 tail hypothesis tests
alpha = 0.025
df = 24
# X² for upper tail
print(f'The critical value X²U for the upper tail is {scipy.stats.chi2.ppf(1-alpha, df=df)}') # 39.3641
# X² for lower tail
print(f'The critical value X²L for the lower tail is {scipy.stats.chi2.ppf(alpha, df=df)}') # 12.4011


# In[52]:


# find Chi-Square critical value for 2 tail hypothesis tests
alpha = 0.05

df = 28
# X² for upper tail
print(f'The critical value X²U for the upper tail is {scipy.stats.chi2.ppf(1-alpha, df=df)}') # 39.3641
# X² for lower tail
print(f'The critical value X²L for the lower tail is {scipy.stats.chi2.ppf(alpha, df=df)}') # 12.4011


# <a id="1"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Confidence Interval for Chi-Square (σ²)</strong></h1>
# 
# ![Confidence Interval for Chi-Square (σ²)](../media/images/CI_variance.png)
# 
# #### A random sample of 23 points is taken from a normally dis- tributed population, find the 95% confidence interval for the unknown population variance σ², where the sample variance s² = 2.3.
# 
# Now for α = 0.05 we want an area of 0.025 in each of the upper and lower tails. Looking up the χ2 distribution with α = 0.025 and df = 23 − 1 = 22 we get χ2U = 36.7807. Similarly looking up the χ2 distribution with α = 0.975 and df =23−1=22 we get χ2L =10.9823.
# Substituting these into the confidence interval gives

# In[43]:





# <a id="1"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Hypothesis Test for a Single Variance</strong></h1>
# 
# The assumption that is made when deriving the X² distribution is that the sample measurements x1, x2, · · · , xn are from a normally distributed popula- tion. This then is an assumption that needs to be verified when we apply this test.
# 
# + H₀: σ² = σo²
# + H₁: σ² # σo²                  | 2 tail test
# + H₁: σ² > σo²                  | upper tail test
# + H₁: σ² < σo²                  | lower tail test
# 
# ![test_statistic_single_variance](../media/images/test_statistic_single_variance.png)

# In[43]:





# <a id="1"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>F Distribution</strong></h1>

# In[43]:





# <a id="1"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Hypothesis Test for Two Variances</strong></h1>

# In[43]:





# # References
# 
# + [Non parametric statistical significance test in Python](https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/)
# + [p value Calculation Python](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)
# + [Chi-Square Distribution Python](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html)
# + [Chi-squared distribution table from Probability and Statistics by Jay L. Devore.]()

# In[43]:




