#!/usr/bin/env python
# coding: utf-8

# + One way Analysis of Variance (ANOVA)
# + Kruskal Wallis test
# + Chi Square Goodness of Fit test
# + Chi Square Tests for Independence / Association Test

# In[19]:


# import libraries
get_ipython().system('pip install researchpy')
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import researchpy as rp
from scipy import stats


# In[36]:


# set the general path of the external df
external_df_path = os.path.join(os.path.pardir, 'data', 'interim')

# set the path for specific dfset from external dfset
df = os.path.join(external_df_path, 'cleaned_data.csv')

# import dfset
df = pd.read_csv(df, delimiter=',', skipinitialspace=True)

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


# # HOMOGENEITY OF VARIANCE
# 
# testing this assumption is the Levene's test of homogeneity of variances. This can be completed using the levene() method from Scipy.stats.

# In[37]:


# stats.levene(*[df['score'][df['region'] == region] for region in df.region.unique()])


# ## -------> OBSERVATION
# The Levene's test of homogeneity of variances is not significant which indicates that the groups have non-statistically significant difference in their varability. Again, it may be worthwhile to check this assumption visually as well.

# ### ------> OBSERVATIONS
# The graphical testing of homogeneity of variances supports the statistical testing findings which is the groups have equal variance.
# 
# By default box plots show the median (orange line in graph above). The green triangle is the mean for each group which was an additional argument that was passed into the method.

# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>One way Analysis of Variance (ANOVA)</strong></h1>
# 
# ## ANOVA Hypotheses
# + Null hypothesis: Groups means are equal (no variation in means of groups)
# H0: μ1=μ2=…=μp
# + Alternative hypothesis: At least, one group mean is different from other groups
# H1: All μ are not equal
# 
# ### Parametric test assumptions
# + Population distributions are normal
# + Samples have equal variances
# + Independence
# 
# ## ANOVA Assumptions
# + Residuals (experimental error) are approximately normally distributed (Shapiro-Wilks test or histogram)homoscedasticity or Homogeneity of variances (variances are equal between treatment groups) (Levene’s, Bartlett’s, or Brown-Forsythe test)
# + Observations are sampled independently from each other (no relation in observations between the groups and within the groups) i.e., each subject should have only one response
# + The dependent variable should be continuous. If the dependent variable is ordinal or rank (e.g. Likert item data), it is more likely to violate the assumptions of normality and homogeneity of variances. If these assumptions are violated, you should consider the non-parametric tests (e.g. Mann-Whitney U test, Kruskal-Wallis test).
# 
# 
# ## How ANOVA works?
# + Check sample sizes: equal number of observation in each group
# + Calculate Mean Square for each group (MS) (SS of group/level-1); level-1 is a degrees of freedom (df) for a group
# + Calculate Mean Square error (MSE) (SS error/df of residuals)
# + Calculate F value (MS of group/MSE)
# + Calculate p value based on F value and degrees of freedom (df)
# 
# ## Questions?
# + Imbalance label problem (unequal sample size for each group) data
# 
# # ANOVA TABLE
# 
# | **ANOVA Source** |   **df**   | **SS** | **MS**          | **F**    | **Notes**           |
# |:----------------:|:----------:|:------:|:---------------:|:--------:|:-------------------:|
# |  **Treatments**  | dfTr = k-1 |  SSTr  | MSTr=SSTr/(k-1) | MSTr/MSE | k: number of groups |
# |    **Errors**    | dfE = n-k  | SSE    | MSE=SSE/(n-k)   |          | n: sample size      |
# |    **Total**     | dfT = n-1  | SST    |                 |          |                     |
# 

# In[ ]:


k = 0
n = 0
dfTr = 0
dfE = 0
dfT = 0
SSTr = 0
SSE = 0
SST = 0
MSTr = 0
MSE = 0
F = 0
p = 0

dfTr = k - 1
dfE = n - k
dfT = n - 1
SSTr = SST - SSE
MSTr = SSTr / dfTr
MSE = SSE / dfE
SST = SSTr + SSE
F = MSTr / MSE
p = stats.f.sf(F, dfTr, dfE)


# In[22]:


n = 18
k = 3
dfT = 17
SSTr = 57.11
SST = 73.75

dfTr = k - 1
dfE = n - k
MSTr = SSTr / dfTr
SSE = SST - SSTr
MSE = SSE / dfE
F = MSTr / MSE
p = stats.f.sf(F, dfTr, dfE)
print("F value: ", F and "p value: ", p)


# # F Critical Value
# 
# `scipy.stats.f.ppf(q, dfn, dfd)`
# 
# where:
# 
# `q`: The significance level to use
# `dfn`: The numerator degrees of freedom
# `dfd`: The denominator degrees of freedom

# 
# 
# 
# Steps in the test:
# 1. First write down the null and alternate hypothesis.
#     >> H0 : μ1 =μ2 =μ3 =···=μk
#     >> H1 : at least one pair have different means.
# 2. Next calculate the test statistic F using the data and present it in a
# table as above.
# 3. Find the Rejection region for the corresponding alternate hypothesis
# and chosen α value, that is find Fα,k−1,n−k. **The rejection area is in the right hand side of the critical value.
# 4. Reject or Don’t Reject If the test statistic F falls in the rejection region, reject H0 and conclude H1 is true, or else do not reject H0. You should also interpret the result in words.

# In[7]:


import scipy.stats

# F0.05,2,12 = 3.89
# sample size
n = 18
# number of groups
k = 3
# significance level
q = 1 - .05

# numerator degrees of freedom
dfn = k - 1
print(f'Numerator df v1: {dfn}')

# denominator degrees of freedom
dfd = n - k
print(f'Denominator df v2: {dfd}')

# F critical value
print(f'The critical value from F-table: F({q}, {k-1}, {n-k}) = {scipy.stats.f.ppf(q, k-1, n-k)}')

#find F critical value
print(f'F critical value: {scipy.stats.f.ppf(q=q, dfn=dfn, dfd=dfd)}')


# In[51]:


import scipy.stats

# F0.05,2,12 = 3.89

performance1 = [8, 9, 11, 10, 11, 9, 10, 10]
performance2 = [13, 10, 9, 10, 11, 8, 14, 13]
performance3 = [10, 12, 13, 10, 11, 11, 13]

# number of values in all groups
n = len(performance1) + len(performance2) + len(performance3)
# number of groups
k = 3
# significance level
q = 1 - .05
# numerator degrees of freedom
dfn = k - 1
# denominator degrees of freedom
dfd = n - k
#find F critical value
print(f'F critical value: {scipy.stats.f.ppf(q=q, dfn=dfn, dfd=dfd)}')


# In[50]:


# Importing library
from scipy.stats import f_oneway

performance1 = [8, 9, 11, 10, 11, 9, 10, 10]
performance2 = [13, 10, 9, 10, 11, 8, 14, 13]
performance3 = [10, 12, 13, 10, 11, 11, 13]

# Conduct the one-way ANOVA
f_oneway(performance1, performance2, performance3)


# ### -------> OBSERVATION
# 
# The purpose of this study was to test for a difference in score between the region. The overall average score was 610.55 95% CI(600.81,620.29) with group averages of.... There is a statistically insignificant difference between the regions and their effects the scores, F= 1.19, p-value= 0.27.
# 
# As the p value (0.27) is insignificant, we fail to reject the null hypothesis and conclude that regions have equal variances.
# 

# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Kruskal Wallis test</strong></h1>
# 
# 
# Assumptions for the test:
# 1. At least one of the two large sample conditions are met.
# 2. All the samples are random samples.
# 3. All the populations being sampled have the same shaped probability density function, with possibly different means.
# 4. The populations are independent.
# 
# H0 : μ1 =μ2 =μ3 =···=μk
# H1 : at least two μi differ.
# 
# **NOTE**: The larger the differences the larger the test statistic H. This is why the test is only an upper tail test.

# In[56]:


# Find the Chi-Square Critical Value
import scipy.stats

# find Chi-Square critical value for 2 tail hypothesis tests
alpha = float(0.01)
k = 4
degree_freedom = k - 1
print(f'degrees of freedom: {(k - 1)}')
# X² for upper tail
print(f'The critical value X²U for the upper tail is {scipy.stats.chi2.ppf(1 - alpha, df=degree_freedom)}')


# In[10]:


# Conduct the Kruskal-Wallis Test
result = stats.kruskal(*[df['score'][df['region'] == region] for region in df.region.unique()])

# Print the result
print(result)


# In[55]:


# group1 = [624, 680, 454, 510, 539]
# group2 = [425, 595, 737, 459, 709, 482]
# group3 = [397, 794, 595, 539, 680, 652]
# group4 = [482, 510, 369, 567, 595]
#
#
# from scipy import stats
#
# #perform Kruskal-Wallis Test
# stats.kruskal(group1, group2, group3, group4)


# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Chi Square Goodness of Fit test</strong></h1>
# 
# The Chi-Square Goodness of fit test is a non-parametric statistical hypothesis test that’s used to determine how considerably the observed value of an event differs from the expected value. it helps us check whether a variable comes from a certain distribution or if a sample represents a population. The observed probability distribution is compared with the expected probability distribution.
# 
# 
# if chi_square_ value > critical value, the null hypothesis is rejected. if chi_square_ value <= critical value, the null hypothesis is accepted.
# 
# 
# H0: (null hypothesis) A variable follows a hypothesized distribution.
# H1: (alternative hypothesis) A variable does not follow a hypothesized distribution.
# 
# `chisquare(f_obs, f_exp)`
# 
# where:
# 
# `f_obs`: An array of observed counts.
# `f_exp`: An array of expected counts. By default, each category is assumed to be equally likely.

# In[11]:


# First, we will create two arrays to hold our observed and expected number of customers for each day:
expected = [50, 50, 50, 50, 50]
observed = [50, 60, 40, 47, 53]

import scipy.stats as stats

#perform Chi-Square Goodness of Fit Test
stats.chisquare(f_obs=observed, f_exp=expected)


# Note that the p-value corresponds to a Chi-Square value with n-1 degrees of freedom (dof), where n is the number of different categories. In this case, dof = 5-1 = 4. You can use the Chi-Square to P Value Calculator to confirm that the p-value that corresponds to X2 = 4.36 with dof = 4 is 0.35947.
# 
# Since the p-value (.35947) is not less than 0.05, we fail to reject the null hypothesis. This means we do not have sufficient evidence to say that the true distribution of customers is different from the distribution that the shop owner claimed.

# A genetics experiment, crossing two types of sorghum, in theory, should produce offspring with the colours red, yellow, and white in the ratio 9:3:4.
# The outcome for 368 experimental plants was 195 red, 73 yellow, and 100 white. Does this data contradict the theory, using α = 0.01?
# The probabilities are: red, 9/16; yellow, 3/16, and white, 4/16. If we let red be outcome 1, yellow be outcome 2, and white outcome 3, we have the following null and alternate hypothesis.

# In[76]:


# importing packages
import scipy.stats as stats

# n = 168
# p1=0.35
# p2=0.35
# p3=0.2
# p4=0.1
# no of hours a student studies
# in a week vs expected no of hours
observed = [27, 31, 25, 17]
# expected = [p1*n, p2*n, p3*n, p4*n]
expected = [15, 21, 25, 39]

degree_freedom = int(len(observed) - 1)
alpha = float(0.01)

# Chi-Square Goodness of Fit Test
chi_square_test_statistic, p_value = stats.chisquare(
    observed, expected)

print(f'Degree of freedom is: {degree_freedom}')
# chi square test statistic and p value
print('chi_square_test_statistic is : ' +
      str(chi_square_test_statistic))
print('p_value : ' + str(p_value))

# find Chi-Square critical value
print(stats.chi2.ppf(1 - alpha, df=degree_freedom))


# In[ ]:





# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Chi Square Test for Independence</strong></h1>
# 
# 
# ### χ^2 test of independence assumptions
# + The two samples are independent
# + No expected cell count is = 0
# + No more than 20% of the cells have and expected cell count < 5
# 
# ![](../media/images/tests_for_Independence.png)
# 
# The null and alternate hypothesis in this problem are,
# H0 : The two factors are independent.
# H1 : the two factors are dependent.

# In[78]:


# Find the Chi-Square Critical Value
import scipy.stats

# find Chi-Square critical value for 2 tail hypothesis tests
alpha = float(0.05)
rows = 2
cols = 5
degree_freedom = (rows - 1) * (cols - 1)
print(f'degrees of freedom: {(rows - 1) * (cols - 1)}')
# X² for upper tail
print(f'The critical value X²U for the upper tail is {scipy.stats.chi2.ppf(1 - alpha, df=degree_freedom)}')


# In[14]:


rp.summary_cat(df[["region", "gender"]])


# The table was called a contingency table, by Karl Pearson, because the intent is to help determine whether one variable is contingent upon or depends upon the other variable.

# In[15]:


import scipy.stats as stats

crosstab = pd.crosstab(df["region"], df["gender"])
stats.chi2_contingency(crosstab)


# In[16]:


crosstab, test_results, expected = rp.crosstab(df["region"], df["gender"],
                                               test="chi-square",
                                               expected_freqs=True,
                                               prop="cell")

crosstab


# In[17]:


test_results


# ### ASSUMPTION CHECK
# 
# Checking the assumptions for the χ2 test of independence is easy. Let's recall what they are:
# 
# + The two samples are independent
# + The variables were collected independently of each other, i.e. the answer from one variable was not dependent on the answer of the other
# + No expected cell count is = 0
# + No more than 20% of the cells have and expected cell count < 5
# The last two assumptions can be checked by looking at the expected frequency table.

# In[18]:


expected


# In[65]:


table = [[95, 55],
         [103, 247]]

table


# In[68]:


# chi-squared test with similar proportions
from scipy.stats import chi2_contingency
from scipy.stats import chi2

# contingency table
table = table
print(table)
stat, p, dof, expected = chi2_contingency(table)
print('\n\n')

print('dof=%d' % dof)
print('\n\n')

print(expected)
# interpret test-statistic
prob = 0.99
critical = chi2.ppf(prob, dof)
print('\n\n')
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
print('\n\n')
if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')
print('\n\n')
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')


# # Summary
# This section lists some ideas for extending the tutorial that you may wish to explore.
# 
# Update the chi-squared test to use your own contingency table.
# Write a function to report on the independence given observations from two categorical variables
# Load a standard machine learning dataset containing categorical variables and report on the independence of each.
# 
# Pairs of categorical variables can be summarized using a contingency table.
# The chi-squared test can compare an observed contingency table to an expected table and determine if the categorical variables are independent.
# How to calculate and interpret the chi-squared test for categorical variables in Python.
# 

# ## --------> OBSERVATION
# The one piece of information that researchpy calculates that scipy.stats does not is a measure of the strength of the relationship - this is akin to a correlation statistic such as Pearson's correlation coefficient. A good peer-reviewed article that is not behind a paywall is written by Akoglu (2018). The following table is reproduced from the mentioned article.

# In[9]:


# The .py format of the jupyter notebook
import os

for fname in os.listdir():
    if fname.endswith('ipynb'):
        os.system(f'jupyter nbconvert {fname} --to python')


# 
# # References
# Seabold, Skipper, and Josef Perktold. “statsmodels: Econometric and statistical modeling with python.” Proceedings of the 9th Python in Science Conference. 2010.
# Virtanen P, Gommers R, Oliphant TE, Haberland M, Reddy T, Cournapeau D, Burovski E, Peterson P, Weckesser W, Bright J, van der Walt SJ. SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature methods. 2020 Mar;17(3):261-72.
# Mangiafico, S.S. 2015. An R Companion for the Handbook of Biological Statistics, version 1.3.2.
# Knief U, Forstmeier W. Violating the normality assumption may be the lesser of two evils. bioRxiv. 2018 Jan 1:498931.
# Kozak M, Piepho HP. What’s normal anyway? Residual plots are more telling than significance tests when checking ANOVA assumptions. Journal of Agronomy and Crop Science. 2018 Feb;204(1):86-98.

# In[19]:




