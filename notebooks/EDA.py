#!/usr/bin/env python
# coding: utf-8

# <h1 style="color:#ffc0cb;font-size:70px;font-family:Georgia;text-align:center;"><strong>Crazy Crossfit in 2015</strong></h1>
# 
# ### 1. Preprocessing
# + Import python libraries and dataset
# + Merge 2 data frames (athletes & leaderboard_15)
# + Put change labels on the data.
# + Check and drop duplicated rows
# + Strip extra white-space and lowercase string content values
# + Make athlete_id unique
# + Drop meaningless columns or columns contains only a value: `retrieved_datetime_x`, `retrieved_datetime_y`, `year`, `stage`, `scaled`, `howlong`
# 
# 
# ### 2. EDA
# created correlation maps to see the relationship between variables and Word Cloud to visualise text contents
# 
# ### 3. Basic Statistics
# + Descriptive Statistics
# + Probability Distributions
# + Normality Test
# + Confidence Intervals
# + Normality Test
# 
# ### 4. Inferential Statistics
# + Hypothesis Testing
# 
# + The Mann-Whitney U test for comparing independent data samples: the nonparametric version of the Student t-test.
# + The Wilcoxon signed-rank test for comparing paired data samples: the nonparametric version of the paired Student t-test.
# + The Kruskal-Wallis H and Friedman tests for comparing more than two data samples: the nonparametric version of the ANOVA and repeated measures ANOVA tests.
# 
# 
# <h1 style="color:#ffc0cb;font-size:70px;font-family:Georgia;text-align:center;"><strong>2. EDA</strong></h1>

# In[48]:


# Install a conda package in the current Jupyter kernel
import sys
get_ipython().system('{sys.executable} -m pip install statsmodels')
get_ipython().system('{sys.executable} -m pip install wordcloud')


# work with df in tabular representation
from datetime import time
import pandas as pd
# round the df in the correlation matrix
import numpy as np
import os
from scipy.stats import t
from scipy import stats
from statistics import *
from scipy.stats import shapiro
from numpy import mean
from numpy import std

# Modules for df visualization
import seaborn as sns
import matplotlib.pyplot as plt

from wordcloud import WordCloud

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


# In[49]:


# check the version of the packages
print("Numpy version: ", np.__version__)
print("Pandas version: ",pd.__version__)
get_ipython().system(' python --version')


# In[50]:


# set the general path of the external df
external_df_path = os.path.join(os.path.pardir,'data','interim')

# set the path for specific dfset from external dfset
df = os.path.join(external_df_path, 'cleaned_data.csv')


# In[51]:


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

# print first 5 rows
df.head(3)


# In[52]:


# see the static of all numerical column
df.describe().T


# In[53]:


df.run5k.unique()


# In[54]:


# Number of Unique Athlete
print(df.athlete_id.nunique() == df.shape[0])
# Summary Stats: weight
df.describe([0.01, 0.05, 0.10, 0.20, 0.80, 0.90, 0.95, 0.99])[["weight"]].T


# In[55]:


# Groups & Target Summary Stats
df.groupby("gender").weight.agg(["count", "median", "mean", "std", "max"])


# In[56]:


def visualize_word(col_name):
    text = df[col_name].values
    wordcloud = WordCloud().generate(str(text))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig('visualize_word.png')
    plt.show()


# In[57]:


visualize_word('eat')


# ### The Central Limit Theorem
# If n>30, the Central Limit Theorem can be used.
# 
# Unlike the normal case, these histograms all differ in shape. In particular, they become progressively less skewed as the sample size n increases.
# 
# provide convincing evidence that a sample size of n=30 is sufficient to overcome the skewness of the population distribution and give an approximately normal X sampling distribution.

# In[58]:


# histogram average weight between athele male and female
import plotly.express as px

df1 = df[['weight','gender']]
fig = px.histogram(df1, x="gender",y="weight",  histfunc='avg')
fig.show()


# <a id="7"></a> <br>
# # Relationship Between Variables

# <a id="8"></a> <br>
# ## Correlation
# * Strength of the relationship between two variables
# * Lets look at correlation between all features.

# In[59]:


f,ax=plt.subplots(figsize = (20,20))
sns.heatmap(df.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Map')
plt.savefig('heatmap to indicates correlation between variables.png')
plt.show()


# ### ----------> OBSERVATION
# 
# + There is a strong positive correlation between `helen` & `fran`, `helen` & `run5k`, `candj` & `backsq`, `backsq` & `deadlift`, `rank` & `score`, `backsq` & `snatch`.
# 
# + There is **multicollinearity** (Mulitple independent variables are highly correlated) between attributes. If I want to feed these feature into my multiple regression model, I would need to drop 1 of the column that is strongly correlated with each other to prevent statistical insignificant problem

# In[60]:


#  Groups & Target Summary Stats
df.groupby("gender").weight.agg(["count", "median", "mean", "std", "max"])


# <a id="1"></a> <br>
# # Histogram
# * How many times each value appears in dfset. This description is called the distribution of variable
# * Most common way to represent distribution of varible is histogram that is graph which shows frequency of each value.
# * Frequency = number of times each value appears

# In[61]:


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


# # ---------
# # Randomly select a 10 number of rows from a dataframe
# 

# In[62]:


df2 = df.sample(n=10, random_state=42)
df2.head(3)


# In[63]:


# convert specified column in the dataframe into series
from numpy import mean
from numpy import std

population_weight = df['weight'].squeeze()
print('Poppulation mean=%.3f stdv=%.3f' % (mean(population_weight), std(population_weight)))
sample_weight = df2['weight'].squeeze()
print('Sample mean=%.3f stdv=%.3f' % (mean(sample_weight), std(sample_weight)))


# In[64]:


# Distribution of population weight
plt.figure(figsize = (20,10))
sns.distplot(population_weight,color = 'pink')
plt.savefig('pop_distplot.png')


# In[65]:


# Distribution of sample weight
plt.figure(figsize = (20,10))
sns.distplot(sample_weight,color = 'pink')
plt.savefig('sample_distplot.png')


# <h1 style="color:#ffc0cb;font-size:70px;font-family:Georgia;text-align:center;"><strong>2. Normality Test</strong></h1>
# 
# I need to decide whether to use parametric or nonparametric statistical methods.
# 
# + H₀: The data is normally distributed.
# + H₁: The data is not normally distributed.
# + H₀: The variances of the samples are the same. 
# + H₁: The variances of the samples are different.
# At α=0.05. If the p-value is >0.05, it can be said that the mean weight is normally distributed.
# 

# # Shapiro-Wilk Test

# In[66]:


def check_normality_shapiro(data):
    test_stat_normality, p_value_normality=stats.shapiro(data)
    print("p value:%.3f" % p_value_normality)
    stat, p = shapiro(df.weight)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    if p_value_normality <0.05:
        print("Reject null hypothesis: The data is not normally distributed")
    else:
        print("Fail to reject null hypothesis: The data is normally distributed")


# In[67]:


check_normality_shapiro(population_weight)


# In[68]:


check_normality_shapiro(sample_weight)


# # Check homogeneity of variance using Levene’s test

# In[69]:


def check_variance_homogeneity(group1, group2):
    test_stat_var, p_value_var= stats.levene(group1,group2)
    print("p value:%.3f" % p_value_var)
    if p_value_var <0.05:
        print("Reject null hypothesis: The variances of the samples are different.")
    else:
        print("Fail to reject null hypothesis: The variances of the samples are same.")


# In[70]:


check_variance_homogeneity(population_weight, sample_weight)


# <h1 style="color:#ffc0cb;font-size:70px;font-family:Georgia;text-align:center;"><strong>3. Parametric Statistical Significance Tests</strong></h1>
# 
# + Ho: Athlete population mean weight is the same as athlete sample mean weight in the CrossFit Game 2015: μ₁=μ₂
# + Ha: The population and sample weights are different μ₁# μ₂
# 
# 
# # T-Sample Test

# In[71]:


ttest,p_value = stats.ttest_ind(np.array(population_weight).astype(int), np.array(sample_weight).astype(int))
print("p value:%.3f" % p_value)
if p_value <0.05:
    print("Reject null hypothesis")
else:
    print("Fail to reject null hypothesis")


# # 1 Small Sample Hypothesis Test
# A test on 10 athletes who competed in the crossfit, the weight of each athlete is inspected, using alpha = 0.05.

# A one-sample t-test checks whether a sample mean differs from the population mean.
# 

# In[72]:


import scipy.stats as stats
import math


# In[73]:


print( population_weight.mean() )
print( sample_weight.mean() )


# In[74]:


# cast population_weight and sample_weight as numeric values
population_weight = population_weight.astype(float)
sample_weight = sample_weight.astype(float)


# In[75]:


stats.ttest_1samp(a = sample_weight,               # Sample data
                 popmean = population_weight.mean())  # Pop mean


# > t critical is: [-2.262,2.262]

# In[76]:


stats.t.ppf(q=0.025,  # Quantile to check
            df=9)  # Degrees of freedom


# In[77]:


stats.t.ppf(q=0.975,  # Quantile to check
            df=9)  # Degrees of freedom


# In this case, the p-value is higher than our significance level α (0.05), so we should reject the null hypothesis. If we were to construct a 95% confidence interval for the sample, it captures a population mean of 180.638

# In[78]:


sigma = sample_weight.std()/math.sqrt(10)  # Sample stdev/sample size

stats.t.interval(0.95,                        # Confidence level
                 df = 9,                     # Degrees of freedom
                 loc = sample_weight.mean(), # Sample mean
                 scale= sigma)                # Standard dev estimate


# On the other hand, it is not significant at the 80% confidence level. This means if we were to construct a 80% confidence interval, it even would not capture the population mean:

# In[79]:


stats.t.interval(alpha = 0.8,                # Confidence level
                 df = 9,                     # Degrees of freedom
                 loc = sample_weight.mean(), # Sample mean
                 scale= sigma)                # Standard dev estimate


# # Calculate Z score for mean weight of different gender

# In[90]:


# selecting rows based on condition
male_weight = df[df['gender'] == 'male']
female_weight = df[df['gender'] == 'female']
df.gender.unique()


# In[81]:


# Import statistics Library
import statistics

print("Population Male Mean: "+str(male_weight['weight'].mean()))
male_weight_sample = male_weight.sample(frac=0.10)
sample_mean_male = male_weight_sample['weight'].mean()
print("Sample Male Mean: "+str(sample_mean_male))
sample_std_male = statistics.stdev(male_weight_sample.weight)
print("Sample Male Standard Deviation: "+str(sample_std_male))


# In[82]:


print("Population female Mean: "+str(female_weight['weight'].mean()))
female_weight_sample = female_weight.sample(frac=0.10)
sample_mean_female = female_weight_sample['weight'].mean()
print("Sample female Mean: "+str(sample_mean_female))
sample_std_female = statistics.stdev(female_weight_sample.weight)
print("Sample female Standard Deviation: "+str(sample_std_female))


# In[83]:


import math
# Confidence Level 95 %  for one sided Normal curve
zscore_critical = 1.65
# Calculate the test statistics
zscore_test_stat = ((sample_mean_male - sample_mean_female)*math.sqrt(8916))/sample_std_female
print(zscore_test_stat)


# <a id="1"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Wilcoxon signed-rank test</strong></h1>
# 
# + Single paired sample.
# + The test assumes that the distribution being sampled is continuous and symmetric about its mean.

# In[91]:


# make the dataframe one-dimensional
male_weight = male_weight['weight']
female_weight = female_weight['weight']


# In[99]:


# Wilcoxon signed-rank test
from scipy.stats import wilcoxon

# generate two paired samples
data1 = female_weight.head(25)
data2 = female_weight.tail(25)
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
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Mann-Whitney U test</strong></h1>
# 
# + 2 independent samples

# In[95]:


# generate two independent samples
data1 = male_weight.sample(25)
data2 = female_weight.sample(25)


# In[96]:


data11 = data1.to_numpy(dtype=int)


# In[97]:


data22 = data2.to_numpy(dtype=int)


# In[98]:


# Mann-Whitney U test
from scipy.stats import mannwhitneyu
# compare samples
stat, p = mannwhitneyu(data11, data22)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')


# <a id="1"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Single Variance</strong></h1>
# 

# In[102]:


# Find the Chi-Square Critical Value
import scipy.stats
# find Chi-Square critical value for 2 tail hypothesis tests
alpha = 0.05

df = 24

# X² for upper tail
print(f'The critical value X²U for the upper tail is {scipy.stats.chi2.ppf(1-alpha, df=df)}') # 39.3641
# X² for lower tail
print(f'The critical value X²L for the lower tail is {scipy.stats.chi2.ppf(alpha, df=df)}') # 12.4011


# In[107]:


# Python Program illustrating
# numpy.var() method
import numpy as np

# 1D array
arr = data2

print("arr : ", arr)
print("var of arr : ", np.var(arr))


# <a id="1"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Summary</strong></h1>

# # References
# 
# + Machine Learning Mastery. 2022. A Gentle Introduction to Normality Tests in Python. [online] Available at: <https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/> [Accessed 22 August 2022].
# 
# + Docs.scipy.org. 2022. Statistical functions (scipy.stats) — SciPy v1.9.0 Manual. [online] Available at: <https://docs.scipy.org/doc/scipy/reference/stats.html> [Accessed 26 August 2022].
# 
# + [Cast Series to Numpy int datatype](https://pandas.pydata.org/pandas-docs/version/0.24.0rc1/api/generated/pandas.Series.to_numpy.html)

# In[88]:





# In[88]:





# In[88]:




