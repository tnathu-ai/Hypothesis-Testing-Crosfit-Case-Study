#!/usr/bin/env python
# coding: utf-8

# In[29]:


# import libraries
get_ipython().system('pip install researchpy')
import os
import numpy as np
import pandas as pd
import researchpy as rp


# In[5]:


# set the general path of the external df
external_df_path = os.path.join(os.path.pardir,'data','interim')

# set the path for specific dfset from external dfset
df = os.path.join(external_df_path, 'cleaned_data.csv')

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


# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>One way Analysis of Variance (ANOVA)</strong></h1>
# 
# ## ANOVA Hypotheses
# + Null hypothesis: Groups means are equal (no variation in means of groups)
# H0: μ1=μ2=…=μp
# + Alternative hypothesis: At least, one group mean is different from other groups
# H1: All μ are not equal
# 
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

# In[15]:


# generate a boxplot to see the data Distribution of scores by region. Using boxplot, we can
# easily detect the differences between different regions
import matplotlib.pyplot as plt
import seaborn as sns
# set with and height of the figure
plt.figure(figsize=(24,8))
ax = sns.boxplot(x='region', y='score', data=df, color='#99c2a2')
ax = sns.swarmplot(x="region", y="score", data=df, color='#7d0013')
# set title with matplotlib
plt.title('Distribution of scores by region')
plt.show()


# In[31]:


rp.summary_cont(df['score'])


# In[30]:


rp.summary_cont(df['score'].groupby(df['region']))


# In[34]:


print(f'NUMBER OF CATEGORIES: {df.region.nunique()}; \n\nUNIQUE NAMES OF THE CATEGORIES {df.region.unique()}\n\n\n')


# In[ ]:


import scipy.stats as stats

"""
UNIQUE NAMES OF THE CATEGORIES <StringArray>
[      'south central',          'south west',        'mid atlantic',
       'north central',          'north east', 'southern california',
              'europe',          'north west', 'northern california',
         'canada east',         'canada west',           'australia',
       'latin america',                'asia',        'central east',
          'south east',              'africa']
Length: 17, dtype: string
"""

# there are 17 regions
stats.f_oneway(df['score'][df['region'] == 'south central'],
               df['score'][df['region'] == 'north central'],
               df['score'][df['region'] == 'europe'],
               df['score'][df['region'] == 'north west'],
               df['score'][df['region'] == 'south west'],
               df['score'][df['region'] == 'mid atlantic'],
               df['score'][df['region'] == 'north east'],
               df['score'][df['region'] == 'southern california'],
               df['score'][df['region'] == 'northern california'],
               df['score'][df['region'] == 'canada east'],
               df['score'][df['region'] == 'canada west'],
               df['score'][df['region'] == 'australia'],
               df['score'][df['region'] == 'latin america'],
               df['score'][df['region'] == 'asia'],
               df['score'][df['region'] == 'central east'],
               df['score'][df['region'] == 'south east'],
               df['score'][df['region'] == 'africa'])

# calculate f_oneway by looping through unique regions
stats.f_oneway(*[df['score'][df['region'] == region] for region in df.region.unique()])



# In[26]:


# if you have a stacked table, you can use bioinfokit v1.0.3 or later for the bartlett's test
from bioinfokit.analys import stat
res = stat()
res.bartlett(df=df, res_var='score', xfac_var='region')
res.bartlett_summary


# In[28]:


# Levene’s test can be used to check the Homogeneity of variances when the data is not drawn from normal distribution.
# if you have a stacked table, you can use bioinfokit v1.0.3 or later for the Levene's test
from bioinfokit.analys import stat
res = stat()
res.levene(df=df, res_var='score', xfac_var='region')
res.levene_summary


# ### -------> OBSERVATION
# As the p value (0.56) is significant, we reject the null hypothesis and conclude that regions have unequal variances.

# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Kruskal Wallis test</strong></h1>

# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Chi Square Goodness of Fit test</strong></h1>

# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Tests for Independence</strong></h1>

# In[ ]:





# In[18]:


# The .py format of the jupyter notebook
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

# In[ ]:




