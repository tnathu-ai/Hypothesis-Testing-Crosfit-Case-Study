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
# 

# <a id="1"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>1. Data Preprocessing</strong></h1>
# 
# <a id="1.1"></a>
# # 1.1 Importing Necessary Libraries and datasets

# In[22]:


# Install a conda package in the current Jupyter kernel
import sys
get_ipython().system('{sys.executable} -m pip install missingno')

# work with data in tabular representation
from datetime import time
import pandas as pd
# round the data in the correlation matrix
import numpy as np
import os

# Modules for data visualization
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

plt.rcParams['figure.figsize'] = [6, 6]

# Ensure plots are embedded within the Jupyter notebook itself. Without this command, sometimes plots may show up in pop-up windows
get_ipython().run_line_magic('matplotlib', 'inline')

# overwrite the style of all the matplotlib graphs
sns.set()

# ignore DeprecationWarning Error Messages
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# check the version of the packages
print("Numpy version: ", np.__version__)
print("Pandas version: ",pd.__version__)
get_ipython().system(' python --version')


# <a id="1.2"></a>
# # Data Retrieving
# ***
# In order to load data properly, the data in csv file have to be examined carefully. First of all, all the categories are seperated by the "," and strip the extra-whitespaces at the begin by setting "skipinitialspace = True".

# In[3]:


# set the general path of the external data
external_data_path = os.path.join(os.path.pardir,'data','external')

# set the path for specific dataset from external dataset
athletes = os.path.join(external_data_path, 'athletes.csv')
leaderboard_15 = os.path.join(external_data_path, 'leaderboard_15.csv')


# In[4]:


# import dataset
athletes = pd.read_csv(athletes, delimiter=',', skipinitialspace = True)
# print dataset info
print("The shape and data type of the ORGINAL data:", str(athletes.info()))
# print first 5 rows
athletes.head(5)


# In[5]:


# import dataset
leaderboard_15 = pd.read_csv(leaderboard_15, delimiter=',', skipinitialspace = True)
# print dataset info
print("The shape and data type of the ORGINAL data:", str(leaderboard_15.info()))
# print first 5 rows
leaderboard_15.head(5)


# In[6]:


df = pd.merge(athletes,leaderboard_15,on='athlete_id')
df.head(3)
df.info()


# In[7]:


def missing_percentage(df):
    """This function takes a DataFrame(df) as input and returns two columns, total missing values and total missing values percentage"""
    total = df.isnull().sum().sort_values(ascending=False)[df.isnull().sum().sort_values(ascending=False) != 0]
    percent = round(df.isnull().sum().sort_values(ascending=False) / len(df) * 100, 2)[
        round(df.isnull().sum().sort_values(ascending=False) / len(df) * 100, 2) != 0]
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

# display missing values in descending
print("Missing values in the dataframe in descending: \n", missing_percentage(df).sort_values(by='Total', ascending=False))

# visualize where the missing values are located
msno.matrix(df, color=(255 / 255, 192 / 255, 203 / 255))
yellow_patch = mpatches.Patch(color='pink', label='present value')
white_patch = mpatches.Patch(color='white', label='absent value')
plt.legend(handles=[yellow_patch, white_patch])
plt.savefig('missing_plot.png')
plt.show()


# <br><br>
# <a id="2.6"></a>
# ## Check data types & Make the data homogeneous
# The dtypes that pandas uses are: `float`, `int`, `bool`, `datetime`, `timedelta`, `category` and `object`. I modify data types in my DataFrames to help me transform them into more meaningful metrics
# 
# + Cast pandas objects to a specified dtype (string)¶
# + Numeric data should have for example the same number of digits after the point.

# In[8]:


print("The shape of the data BEFORE CONVERT is (row, column):", str(df.shape))
print("The data types BEFORE CONVERT are:", df.dtypes, "\n\n")

# convert columns to the best possible dtypes, object->string
df = df.convert_dtypes()

# select numeric columns
df_numeric = df.select_dtypes(include=[np.number]).columns.to_list()

# select non-numeric columns
df_string = df.select_dtypes(include='string').columns.tolist()

# print number of numeric column
print("Length of numeric columns: ", len(df_numeric))
print("Length of categorical columns: ", len(df_string))

print("Numeric columns: ", df_numeric)
print("String columns: ", df_string, "\n\n")

print("The shape of the data AFTER CONVERT is (row, column):", str(df.shape))
print("The data types AFTER CONVERT are:", df.dtypes, "\n\n")


# ### Lower Case the content
# In this section we will convert all the string value in the column to lowercase for further processing and keep all the string uniformly format. This will improve the analysis of the data, and also easier to perform any function related to the string.

# In[9]:


# Cast all values inside the dataframe (except the columns' name) into lower case.
df = df.applymap(lambda s: s.lower() if type(s) == str else s)
df.head(3)


# <br><br>
# <a id="2.8"></a>
# ## Sanity checks
# Design and run a small test-suite, consisting of a series of sanity checks to test for the presence of **impossible values** and **outliers** for each attribute.
# ### Check duplication
# + Use the pandas function `.drop_duplicates()` to remove copied rows from a DataFrame

# In[10]:


print(df.duplicated().sum())
df = df.drop_duplicates()
print(df.duplicated().sum())


# In[11]:


# find duplicated athletes id and only keep the last one
# Use the keep parameter to consider only the first instance of a duplicate row to be unique
print(f'Number of duplicated athlete id before: {df.duplicated(subset="athlete_id").sum()}\n')

bool_series = df.duplicated(subset='athlete_id')
print('DataFrame after removing duplicates found in the Name column:')
df = df[~bool_series]
print(f'Number of duplicated athlete id after: {df.duplicated(subset="athlete_id").sum()}\n')
df.info()


# In[12]:


# print out list of types
print(f'NUMBER OF CATEGORIES: {df.year.nunique()}; \n\nUNIQUE NAMES OF THE CATEGORIES {df.year.unique()}\n\n\n')


# In[13]:


# print out list of types
print(f'NUMBER OF CATEGORIES: {df.division.nunique()}; \n\nUNIQUE NAMES OF THE CATEGORIES {df.division.unique()}\n\n\n')
# Convert "division" from int to string
df = df.astype({'division':'string'})
df['division'] = df['division'].str.replace(r'1', 'male').replace(r'2', 'female')

print(f'NUMBER OF CATEGORIES after: {df.division.nunique()}; \n\nUNIQUE NAMES OF THE CATEGORIES {df.division.unique()}\n\n\n')


# In[14]:


# remove pipe?
df['howlong'] = df['howlong'].str.replace(r'\|', '')


# In[15]:


# print out list of types
print(f'NUMBER OF CATEGORIES: {df.howlong.nunique()}; \n\nUNIQUE NAMES OF THE CATEGORIES {df.howlong.unique()}\n\n\n')


# In[16]:


# print out list of types
print(f'NUMBER OF CATEGORIES: {df.stage.nunique()}; \n\nUNIQUE NAMES OF THE CATEGORIES {df.stage.unique()}\n\n\n')


# In[17]:


# print out list of types
print(f'NUMBER OF CATEGORIES: {df.scaled.nunique()}; \n\nUNIQUE NAMES OF THE CATEGORIES {df.scaled.unique()}\n\n\n')


# In[18]:


# Drop non-needed columns
df = df.drop(['retrieved_datetime_x', 'retrieved_datetime_y', 'year', 'stage', 'scaled', 'howlong'], axis=1)


# In[19]:


df.dropna(axis='rows', inplace=True)


# In[20]:


print(df.info())
df.head(5)


# <a id="2.11"></a>
# # Save the Intermediate data that has been transformed

# In[21]:


# set the path of the cleaned data to data and dash
interim_data_path = os.path.join(os.path.pardir,'data','interim')
write_interim_path = os.path.join(interim_data_path, 'cleaned_data.csv')

# To write the data from the data frame into a file, use the to_csv function.
df.to_csv(write_interim_path, index=False)

print("Cleaned data was successfully saved!")


# <a id="5"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Summary</strong></h1>
# 
# Of the 423006 athletes who competed in CrossFit 2015, only 991 entries are valid fill all the reps for each exercise in CrossFit Game 2015. Over the course of the competition's five weeks, a constant attrition rate is affected by a variety of factors, including logistical ones (travel), physical ones (injuries), psychological ones (competition weariness), and trivial ones (forgetting to input score).

# <a id="5"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>References</strong></h1>
# 
# [1]"CrossFit Data - dataset by bgadoci", Data.world, 2022. [Online]. Available: https://data.world/bgadoci/crossfit-data. [Accessed: 29- Jul- 2022].
# 
# [2]S. Swift, "CrossFit Games Data 2012-2015", Sam Swift, 2022. [Online]. Available: CrossFit Games Data 2012-2015. [Accessed: 29- Jul- 2022].

# In[ ]:




