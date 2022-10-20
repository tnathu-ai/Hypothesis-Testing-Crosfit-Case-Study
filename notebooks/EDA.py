#!/usr/bin/env python
# coding: utf-8

# <h1 style="color:#ffc0cb;font-size:70px;font-family:Georgia;text-align:center;"><strong>Crazy Crossfit in 2015</strong></h1>
# 
# ## Intro
# 
# We, as a group, are interested to see patterns, trends, and insights from Crossfit competition statistics in 2015.
# 
# The data has been preprocessed to achieve the highest results and then applied to different analyses and hypothesis tests to identify potential trends and correlations. The dataset we decided to analyse consists of 17 numerical and 12 categorical values with dimensions of 991 x 29.
# 
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
# ### 2. Exploratory data analysis (EDA)
# created correlation maps to see the relationship between variables and Word Cloud to visualise text contents
# 
# ### 3. Basic Statistics
# + Descriptive Statistics
# + Probability Distributions
# + Normality Test
# + Confidence Intervals
# + Normality Test
# 
# ### 4. Normality test
# 
# ### 5. Homogeneity of variance test
# 
# ### 6. Inferential Statistics
# + t test 
# + z test
# + The Mann-Whitney U test for comparing independent data samples
# + The Wilcoxon signed-rank test for comparing paired data samples
# + ANOVA
# + The Kruskal-Wallis H 
# + Independence
# 
# ![tree map for summary](../media/images/hypothesis-testing.png)
# 
# ### 7. Regression Analysis (Refer to the `regression.ipynb` notebook)
# + Simple linear regression
# + Multiple linear regression
# + Quadratic linear regression
# 
# 
# <h1 style="color:#ffc0cb;font-size:70px;font-family:Georgia;text-align:center;"><strong>2. EDA</strong></h1>

# In[2]:


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

# import libraries
get_ipython().system('pip install researchpy')
import os
import numpy as np
import pandas as pd
import researchpy as rp
from scipy.stats import stats
import matplotlib.pyplot as plt

# Ensure that our plots are shown and embedded within the Jupyter notebook itself. Without this command, sometimes plots may show up in pop-up windows
get_ipython().run_line_magic('matplotlib', 'inline')

# overwrite the style of all the matplotlib graphs
sns.set()

# ignore DeprecationWarning Error Messages
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# check the version of the packages
print("Numpy version: ", np.__version__)
print("Pandas version: ",pd.__version__)
get_ipython().system(' python --version')


# In[4]:


# set the general path of the external df
external_df_path = os.path.join(os.path.pardir,'data','interim')

# set the path for specific dfset from external dfset
df = os.path.join(external_df_path, 'cleaned_data.csv')


# In[5]:


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


# In[6]:


# see the static of all numerical column
df.describe().T


# In[7]:


# Number of Unique Athlete
print(df.athlete_id.nunique() == df.shape[0])
# Summary Stats: weight
df.describe([0.01, 0.05, 0.10, 0.20, 0.80, 0.90, 0.95, 0.99])[["weight"]].T


# In[8]:


# Groups & Target Summary Stats
df.groupby("gender").weight.agg(["count", "median", "mean", "std", "max"])


# In[9]:


def visualize_word(col_name):
    text = df[col_name].values
    wordcloud = WordCloud().generate(str(text))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig('visualize_word.png')
    plt.show()


# In[10]:


visualize_word('eat')


# In[11]:


# histogram average weight between athele male and female
import plotly.express as px

df1 = df[['weight','gender']]
fig = px.histogram(df1, x="gender",y="weight",  histfunc='avg')
plt.savefig('hist.png')
fig.show()


# <a id="7"></a> <br>
# # Relationship Between Variables

# <a id="8"></a> <br>
# ## Pearson Correlation
# * Strength of the relationship between two variables
# * Lets look at correlation between all features.

# In[12]:


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

# In[13]:


#  Groups & Target Summary Stats
df.groupby("gender").weight.agg(["count", "median", "mean", "std", "max"])


# <a id="1"></a> <br>
# # Histogram
# * How many times each value appears in dfset. This description is called the distribution of variable
# * Most common way to represent distribution of varible is histogram that is graph which shows frequency of each value.
# * Frequency = number of times each value appears

# In[17]:


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

most_frequent_female_weight_mean = female[1][index_frequent_weight_mean]
most_frequent_male_weight_mean = male[1][index_frequent_weight_mean]

print("Most frequent female weight mean is: ",most_frequent_female_weight_mean)
print("Most frequent male weight mean is: ",most_frequent_male_weight_mean)


# ### ----------> OBSERVATIONS:
#  + We can see that the mean of the male weight is more to the right of the graph which indicates male weight is larger in general. The distribution of the 2 graphs are approximately normally distributed

# # ---------
# ### Randomly select a 10 number of rows from a dataframe
# 

# In[18]:


df2 = df.sample(n=10, random_state=42)
df2.head(3)


# In[20]:


# convert specified column in the dataframe into series
from numpy import mean
from numpy import std

population_weight = df['weight'].squeeze()
print('Population mean=%.3f stdv=%.3f' % (mean(population_weight), std(population_weight)))
sample_weight = df2['weight'].squeeze()
print('Sample mean=%.3f stdv=%.3f' % (mean(sample_weight), std(sample_weight)))


# In[21]:


# Distribution of population weight
plt.figure(figsize = (20,10))
sns.distplot(population_weight,color = 'pink')
plt.savefig('pop_distplot.png')


# In[17]:


# Distribution of sample weight
plt.figure(figsize = (20,10))
sns.distplot(sample_weight,color = 'pink')
plt.savefig('sample_distplot.png')


# ### --------> OBSERVATIONS:
# 
# The distribution of the population weight and sample weight somewhat normally distributed

# <h1 style="color:#ffc0cb;font-size:70px;font-family:Georgia;text-align:center;"><strong>2. Normality Test</strong></h1>
# 
# I need to decide whether to use parametric or nonparametric statistical methods.
# 
# ### The Central Limit Theorem
# If n>30, the Central Limit Theorem can be used.
# 
# Unlike the normal case, these histograms all differ in shape. In particular, they become progressively less skewed as the sample size n increases.
# 
# provide convincing evidence that a sample size of n=30 is sufficient to overcome the skewness of the population distribution and give an approximately normal X sampling distribution.
# 
# + H₀: The data is normally distributed.
# + H₁: The data is not normally distributed.
# + H₀: The variances of the samples are the same. 
# + H₁: The variances of the samples are different.
# At α=0.05. If the p-value is >0.05, it can be said that the mean weight is normally distributed.
# 

# In[22]:


from scipy import stats
from scipy.stats import norm, skew #for some statistics


weight = np.array(df['weight'], dtype=float)
sns.distplot(weight , fit=norm,color = 'pink');

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(weight)
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Weight distribution')

# Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(weight, plot=plt)
plt.show()


# # Shapiro-Wilk Test

# In[64]:


def check_normality_ShapiroWilk(data, data_name):
    test_stat_normality, p_value_normality=stats.shapiro(data)
    print("p value:%.3f" % p_value_normality)
    stat, p = shapiro(df.weight)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    alpha_value = 0.05
    print(f"The distribution for the {data_name} weight: ")
    if p_value_normality < alpha_value:
        print(f"The p value is less than alpha {alpha_value} which p is significant -> Reject null hypothesis: The data is NOT normally distributed")
    else:
        print(f"The p value is larger than alpha {alpha_value} which p is not significant -> Fail to reject null hypothesis: The data is normally distributed")
    print(f"\n\n")


# In[65]:


check_normality_ShapiroWilk(population_weight, "population_weight")

check_normality_ShapiroWilk(sample_weight, "10 samples")

check_normality_ShapiroWilk(df[df["gender"] == "female"].weight, "female weight")

check_normality_ShapiroWilk(df[df["gender"] == "male"].weight, "male weight")


# ### -------> OBSERVATIONS:
# 
# 
# According to the Shapiro-Wilk Test using the p values, the overall population and male are not normally distributed; the female and 10 samples are normally distributed

# ### --------> OBSERVATIONS:
# 
# The distribution graphs (displot, Q-Q plot, histogram) show that the weight distribution is normally distributed

# # Check homogeneity of variance using Levene’s test

# In[66]:


def check_variance_homogeneity_Levene(group1, group2, group1_name, group2_name):
    test_stat_var, p_value_var= stats.levene(group1,group2)
    print("p value:%.3f" % p_value_var)
    alpha_value = 0.05
    print(f"Check homogeneity of variance using Levene’s test between {group1_name} and {group2_name}: ")
    if p_value_var <alpha_value:
        print(f"The p value is less than alpha {alpha_value} which p is significant -> Reject the null hypothesis. The variances of the samples are DIFFERENT because the groups have statistically significant difference in their variability.\n\n")
    else:
        print(f"The p value is larger than alpha {alpha_value} which p is not significant -> Fail to reject the null hypothesis. The variances of the samples are SAME because the groups have non-statistically significant difference in their variability.\n\n")

check_variance_homogeneity_Levene(population_weight, sample_weight, "population weight", "10 samples' weight")


# ## -------> OBSERVATION
# he variances of the samples are SAME

# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>3. Parametric Tests</strong></h1>
# 
# + Ho: Athlete population mean weight is the same as athlete sample mean weight in the CrossFit Game 2015: μ₁=μ₂
# + Ha: The population and sample weights are different μ₁# μ₂
# 
# 
# # T-Sample Test

# In[23]:


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

# In[24]:


import scipy.stats as stats
import math


# In[25]:


print( population_weight.mean() )
print( sample_weight.mean() )


# In[26]:


# cast population_weight and sample_weight as numeric values
population_weight = population_weight.astype(float)
sample_weight = sample_weight.astype(float)


# In[27]:


stats.ttest_1samp(a = sample_weight,               # Sample data
                 popmean = population_weight.mean())  # Pop mean


# # Find T Critical to the Rejection region

# In[70]:


lower_tail_quantile_to_check = 0.025
upper_tail_quantile_to_check = 0.975
degree_of_freedom = 9

lower_tail = stats.t.ppf(q=lower_tail_quantile_to_check,  # Quantile to check
            df=degree_of_freedom)  # Degrees of freedom

upper_tail = stats.t.ppf(q=upper_tail_quantile_to_check,  # Quantile to check
                         df=degree_of_freedom)  # Degrees of freedom


print(f"The Lower Tail rejection region is: (-∞, {lower_tail}]"
      f"\nThe Upper Tail rejection region is: [{upper_tail}, +∞)"
      f"\nThe Two Tail rejection region is: (-∞, {lower_tail}] U [{upper_tail}, +∞)")


# In this case, the p-value is higher than our significance level α (0.05), so we should reject the null hypothesis. If we were to construct a 95% confidence interval for the sample, it captures a population mean of 180.638

# In[30]:


sigma = sample_weight.std()/math.sqrt(10)  # Sample stdev/sample size

stats.t.interval(0.95,                        # Confidence level
                 df = 9,                     # Degrees of freedom
                 loc = sample_weight.mean(), # Sample mean
                 scale= sigma)                # Standard dev estimate


# On the other hand, it is not significant at the 80% confidence level. This means if we were to construct a 80% confidence interval, it even would not capture the population mean:

# In[31]:


stats.t.interval(alpha = 0.8,                # Confidence level
                 df = 9,                     # Degrees of freedom
                 loc = sample_weight.mean(), # Sample mean
                 scale= sigma)                # Standard dev estimate


# # Calculate Z score for mean weight of different gender

# In[32]:


# selecting rows based on condition
male_weight = df[df['gender'] == 'male']
female_weight = df[df['gender'] == 'female']
df.gender.unique()


# In[33]:


# Import statistics Library
import statistics

print("Population Male Mean: "+str(male_weight['weight'].mean()))
male_weight_sample = male_weight.sample(frac=0.10)
sample_mean_male = male_weight_sample['weight'].mean()
print("Sample Male Mean: "+str(sample_mean_male))
sample_std_male = statistics.stdev(male_weight_sample.weight)
print("Sample Male Standard Deviation: "+str(sample_std_male))


# In[34]:


print("Population female Mean: "+str(female_weight['weight'].mean()))
female_weight_sample = female_weight.sample(frac=0.10)
sample_mean_female = female_weight_sample['weight'].mean()
print("Sample female Mean: "+str(sample_mean_female))
sample_std_female = statistics.stdev(female_weight_sample.weight)
print("Sample female Standard Deviation: "+str(sample_std_female))


# In[35]:


import math
# Confidence Level 95 %  for one sided Normal curve
zscore_critical = 1.65
# Calculate the test statistics
zscore_test_stat = ((sample_mean_male - sample_mean_female)*math.sqrt(8916))/sample_std_female
print(zscore_test_stat)


# In[ ]:


# make the dataframe one-dimensional
male_weight = male_weight['weight']
female_weight = female_weight['weight']


# <a id="1"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Single Variance</strong></h1>
# 

# In[ ]:


# # generate two independent samples
# sample_male_data = male_weight.sample(25).to_numpy(dtype=int)
# sample_female_data = female_weight.sample(25).to_numpy(dtype=int)


# # Confidence Invertal

# In[60]:


# Find the Chi-Square Critical Value
import scipy.stats
# find Chi-Square critical value for 2 tail hypothesis tests
alpha = 0.05

# degree of freemdom
dof = len(df)-1

# X² for upper tail
print(f'The critical value X²U for the upper tail is {scipy.stats.chi2.ppf(1-alpha, df=dof)}') # 39.3641
# X² for lower tail
print(f'The critical value X²L for the lower tail is {scipy.stats.chi2.ppf(alpha, df=dof)}') # 12.4011


# In[43]:


data = [*[df['score'][df['region'] == region] for region in df.region.unique()]]

fig = plt.figure(figsize= (30, 15))
ax = fig.add_subplot(111)
ax.set_title("Box Plot of Score by Regions", fontsize= 40)
ax.set

ax.boxplot(data,
           labels= [region for region in df.region.unique()],
           showmeans= True)

plt.xlabel("Regions", fontsize= 30)
plt.ylabel("Score", fontsize= 30)
# bolden the labels
plt.xticks(fontweight= 'bold')
plt.yticks(fontweight= 'bold')

plt.show()


# ### ------> OBSERVATIONS
# The graphical testing of homogeneity of variances supports the statistical testing findings which is the groups have approximately equal variance.
# 
# By default box plots show the median (orange line in graph above). The green triangle is the mean for each group which was an additional argument that was passed into the method.

# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>One way Analysis of Variance (ANOVA)</strong></h1>
# 
# ## ANOVA Hypotheses
# + H0: μ1=μ2=…=μp
# + H1: All μ are not equal
# 
# ### Parametric test assumptions
# 1. The k samples are random.
# 2. All of the k populations being sampled are normally distributed.
# 3. All of the populations are independent.
# 4. All of the k populations have the same variance σ2.
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
# | **ANOVA Source** | **df** | **SS** |     **MS**      | **F**    | **Notes**           |
# |:----------------:|:------:|:------:|:---------------:|:--------:|:-------------------:|
# |  **Treatments**  |  k-1   | SSTr   | MSTr=SSTr/(k-1) | MSTr/MSE | k: number of groups |
# |    **Errors**    |  n-k   | SSE    | MSE=SSE/(n-k)   |          | n: sample size      |
# |    **Total**     |  n-1   | SST    |                 |          |                     |
# 

# In[44]:


# generate a boxplot to see the data Distribution of scores by region. Using boxplot, we can
# easily detect the differences between different regions
import matplotlib.pyplot as plt
import seaborn as sns
# set with and height of the figure
plt.figure(figsize=(24,8))
ax = sns.boxplot(x='region', y='score', data=df, color='#EEC0CB')
ax = sns.swarmplot(x="region", y="score", data=df, color='#7d0013')

# set title with matplotlib
plt.title('Distribution of scores by region')
plt.show()


# In[45]:


rp.summary_cont(df['score'])


# In[46]:


df['score'] = np.array(df['score'],dtype='float64')

rp.summary_cont(df['score'].groupby(df['region']))


# In[47]:


print(f'NUMBER OF CATEGORIES: {df.region.nunique()}; \n\nUNIQUE NAMES OF THE CATEGORIES {df.region.unique()}\n\n\n')


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
# and chosen α value, that is find Fα,k−1,n−k.
# 4. Reject or Don’t Reject If the test statistic F falls in the rejection region, reject H0 and conclude H1 is true, or else do not reject H0. You should also interpret the result in words.

# In[48]:


# ONE-WAY ANOVA USING SCIPY.STATS

# calculate f_oneway by looping through unique regions
stats.f_oneway(*[df['score'][df['region'] == region] for region in df.region.unique()])


# ### ----------> OBSERVATION:
# F statistic is higher than our alpha => we fail to reject the H0

# In[100]:


import scipy.stats

data1 = df.weight.sample(25).to_numpy(dtype=int)
data2 = df.score.sample(25).to_numpy(dtype=int)
data3 = df.height.sample(25).to_numpy(dtype=int)

# number of values in all groups
n = len(data1) + len(data2) + len(data3)

# number of groups
k = 3

# significance level
q = 1-.05
# numerator degrees of freedom
dfn = k-1
# denominator degrees of freedom
dfd = n-k

# print out number of gorjp
print(f'Number of values in all {k} groups: n={n}')

#find F critical value
print(f'F critical value: {scipy.stats.f.ppf(q=q, dfn=dfn, dfd=dfd)}')


# In[101]:


from scipy.stats import f_oneway

# Conduct the one-way ANOVA
f_oneway(data1, data2, data3)


# ### -------> OBSERVATION
# 
# The purpose of this study was to test for a difference in score between the region. The overall average score was 189.291 95%. There is non statistically significant difference between the score, height, weight and their effects the scores, F= 189.291, p-value=2.1288517462948006e-29.
# 
# As the p value significant, we reject the null hypothesis and conclude that they do not have equal variances.
# 

# <h1 style="color:#ffc0cb;font-size:60px;font-family:Georgia;text-align:center;"><strong>4. Non-Parametric Tests</strong></h1>
# 
# + Ho: Athlete population mean weight is the same as athlete sample mean weight in the CrossFit Game 2015: μ₁=μ₂
# + Ha: The population and sample weights are different μ₁# μ₂
# 
# 
# <a id="1"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Wilcoxon signed-rank test</strong></h1>
# 
# + Single paired sample.
# + The test assumes that the distribution being sampled is continuous and symmetric about its mean.

# In[37]:


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
    print('Same distribution. Therefore, we fail to reject the H0)')
else:
    print('Different distribution. Therefore, we reject the H0)')


# <a id="1"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Mann-Whitney U test <br>(Wilcoxon Rank-Sum)</strong></h1>
# 
# + 2 independent population
# + non-parametric
# 
# ### Null Hypothesis
# > + H0 : μX − μY = 0
# > + Ha : μ X − μ Y # 0

# In[58]:


# generate two independent samples
sample_male_data = male_weight.sample(25).to_numpy(dtype=int)
sample_female_data = female_weight.sample(25).to_numpy(dtype=int)


# In[41]:


# Mann-Whitney U test
from scipy.stats import mannwhitneyu
# compare samples
stat, p = mannwhitneyu(sample_male_data, sample_female_data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Same distribution. Therefore, we fail to reject the H0)')
else:
    print('Different distribution. Therefore, we reject the H0)')


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

# In[52]:


# Find the Chi-Square Critical Value
import scipy.stats
# find Chi-Square critical value for 2 tail hypothesis tests
alpha = float(0.01)
k = 4
degree_freedom = k-1
print(f'degrees of freedom: {(k-1)}')
# X² for upper tail
print(f'The critical value X²U for the upper tail is {scipy.stats.chi2.ppf(1-alpha, df=degree_freedom)}')


# In[53]:


# Conduct the Kruskal-Wallis Test
result = stats.kruskal(*[df['score'][df['region'] == region] for region in df.region.unique()])

# Print the result
print(result)


# In[54]:


group1 = [624, 680, 454, 510, 539]
group2 = [425, 595, 737, 459, 709, 482]
group3 = [397, 794, 595, 539, 680, 652]
group4 = [482, 510, 369, 567, 595]


from scipy import stats

#perform Kruskal-Wallis Test
stats.kruskal(group1, group2, group3, group4)


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

# Note that the p-value corresponds to a Chi-Square value with n-1 degrees of freedom (dof), where n is the number of different categories. In this case, dof = 5-1 = 4. You can use the Chi-Square to P Value Calculator to confirm that the p-value that corresponds to X2 = 4.36 with dof = 4 is 0.35947.
# 
# Since the p-value (.35947) is not less than 0.05, we fail to reject the null hypothesis. This means we do not have sufficient evidence to say that the true distribution of customers is different from the distribution that the shop owner claimed.

# In[71]:


# importing packages
import scipy.stats as stats
import numpy as np
n = 168
p1=0.35
p2=0.35
p3=0.2
p4=0.1

observed = [27,31,25,17]
# expected = [p1*n, p2*n, p3*n, p4*n]
expected = [15,21,25,39]

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
print(stats.chi2.ppf(1-alpha, df=degree_freedom))


# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Chi Square Test for Independence</strong></h1>
# 
# 
# ### χ^2 test of independence assumptions
# + The two samples are independent
# + No expected cell count is = 0
# + No more than 20% of the cells have and expected cell count < 5
# 
# <!-- ![](../media/images/tests_for_Independence.png) -->
# 
# The null and alternate hypothesis in this problem are,
# H0 : The two factors are independent.
# H1 : the two factors are dependent.

# In[62]:


# Find the Chi-Square Critical Value
import scipy.stats

# find Chi-Square critical value for 2 tail hypothesis tests
alpha = float(0.05)
rows = 991
cols = 2
degree_freedom = (rows - 1) * (cols - 1)
print(f'degrees of freedom: {(rows - 1) * (cols - 1)}')
# X² for upper tail
print(f'The critical value X²U for the upper tail is {scipy.stats.chi2.ppf(1-alpha, df=degree_freedom)}')


# In[63]:


df[["region", "gender"]]


# In[64]:


rp.summary_cat(df[["region", "gender"]])


# The table was called a contingency table, by Karl Pearson, because the intent is to help determine whether one variable is contingent upon or depends upon the other variable.

# In[65]:


import scipy.stats as stats

crosstab = pd.crosstab(df["region"], df["gender"])
stats.chi2_contingency(crosstab)


# In[66]:


crosstab, test_results, expected = rp.crosstab(df["region"], df["gender"],
                                               test= "chi-square",
                                               expected_freqs= True,
                                               prop= "cell")

crosstab


# In[67]:


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

# In[68]:


expected


# # Summary
# 
# Pairs of categorical variables can be summarized using a contingency table.
# The chi-squared test can compare an observed contingency table to an expected table and determine if the categorical variables are independent.
# How to calculate and interpret the chi-squared test for categorical variables in Python.
# 

# <a id="1"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Summary</strong></h1>

# # References
# 
# + Machine Learning Mastery. 2022. A Gentle Introduction to Normality Tests in Python. [online] Available at: <https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/> [Accessed 22 August 2022].
# 
# + Docs.scipy.org. 2022. Statistical functions (scipy.stats) — SciPy v1.9.0 Manual. [online] Available at: <https://docs.scipy.org/doc/scipy/reference/stats.html> [Accessed 26 August 2022].
# 
# + [Cast Series to Numpy int datatype](https://pandas.pydata.org/pandas-docs/version/0.24.0rc1/api/generated/pandas.Series.to_numpy.html)
