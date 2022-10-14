#!/usr/bin/env python
# coding: utf-8

# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Assignment 2: Data Modelling</strong></h1>
# 

# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Table of Content</strong></h1>
# <br>
# 
# ### 3. [Feature Engineering](#3)
# 3.1 [Data Correlation - Quantify the association of features and accidents](#3.1)
# 
# 3.2 [Gaussian Distributions - Box Cox OR Log Transformation of skewed features](#3.2)
# 
# 3.3 [Assumptions of Regression](#3.3)
# 
# 3.4 [Multicollinearity of Features](#3.4)
# 
# 3.5 [Drop multicollinearity features and high p-value](#3.5)
# 
# 3.6 [Encoding](#3.6)
# 
# 3.7 [Check OLS stats model - Multivariate - Interpretation of the Model Coefficient, the P-value, the R-squared](#3.7)
# 
# 3.8 [Train - Test - Validation Sets](#3.8)
# 
# 3.9 [Feature scaling](#3.9)
# 
# <br>
# 
# ### 4. [Model training](#4)
# 4.1 [Linear Regression](#4.1)
# 
# 4.2 [Regularization Techniques](#4.2)
# + [4.2.1 Ridge Regression](#4.2.1)
# + [4.2.2 Lasso Regression](#4.2.2)
# 
# 4.3 [Polynomial Regression](#4.3)
# 4.4 [Stochastic Gradient Descent](#4.4)
# 4.5 [Artificial Neural Network](#4.5)
# 4.6 [Other Neural Network models](#4.6)
# 4.7 [Random Forest Regressor](#4.7)
# 
# <br>
# 
# ### 5. [Model comparison and export](#5)
# 
# <br>
# 
# ### 6. [Summary](#6)
# 
# <br>
# 
# ### 7. [References](#7)
# 
# <br>
# 
# ### 8. [Appendix](#8)
# 
# <hr>

# <a id="1.2"></a>
# # 1.2 Required  Libraries

# In[3]:


# Install a conda package in the current Jupyter kernel
import sys
get_ipython().system('{sys.executable} -m pip install missingno')

import pandas as pd
import numpy as np

# Modules for data visualization
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.stats import skew  # for some statistics
import matplotlib.style as style

import os

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

plt.rcParams['figure.figsize'] = [6, 6]

# Ensure that our plots are shown and embedded within the Jupyter notebook itself. Without this command, sometimes plots may show up in pop-up windows
get_ipython().run_line_magic('matplotlib', 'inline')

# overwrite the style of all the matplotlib graphs
sns.set()

# ignore DeprecationWarning Eror Messages
import warnings

warnings.filterwarnings('ignore')


# In[4]:


import sklearn

# check the version of the package
print(sklearn.__version__)
print(np.__version__)
print(pd.__version__)


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

# print first 5 rows
df.head(3)


# <a id="1"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>3. Feature Engineering</strong></h1>
# 
# ### Missing Values

# In[39]:


print("Total missing values in TRAIN:", train.isna().sum().sum())
print("Total missing values in TEST:", test.isna().sum().sum())
print("Total missing values in VALIDATION:", validation.isna().sum().sum())


# <a id="3.1"></a>
# ### 3.1 Data Correlation - Quantify the association of features and accidents
# 
# To quantify the pairwise relationships, I compute the Pearson correlation coefficient matrix. 
# 
# 0.2 = weak;
# 0.5 = medium;
# 0.8 = strong;
# 0.9 = very strong

# In[43]:


print (f"Train has {train.shape[0]} rows and {train.shape[1]} columns")
print (f"Test has {test.shape[0]} rows and {test.shape[1]} columns")
print (f"Validation has {validation.shape[0]} rows and {validation.shape[1]} columns")


# In[44]:


# gives us statistical info about the numerical variables. 
train.describe().T


# In[45]:


def plotting_3_chart(df, feature):
    ## Importing seaborn, matplotlab and scipy modules. 
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy import stats
    import matplotlib.style as style
    style.use('fivethirtyeight')

    ## Creating a customized chart. and giving in figsize and everything. 
    fig = plt.figure(constrained_layout=True, figsize=(12,8))
    ## creating a grid of 3 cols and 3 rows. 
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    #gs = fig3.add_gridspec(3, 3)

    ## Customizing the histogram grid. 
    ax1 = fig.add_subplot(grid[0, :2])
    ## Set the title. 
    ax1.set_title('Histogram')
    ## plot the histogram. 
    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1)

    # customizing the QQ_plot. 
    ax2 = fig.add_subplot(grid[1, :2])
    ## Set the title. 
    ax2.set_title('QQ_plot')
    ## Plotting the QQ_Plot. 
    stats.probplot(df.loc[:,feature], plot = ax2)

    ## Customizing the Box Plot. 
    ax3 = fig.add_subplot(grid[:, 2])
    ## Set title. 
    ax3.set_title('Box Plot')
    ## Plotting the box plot. 
    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );
    
plotting_3_chart(train, 'Severity')


# These **three** charts above can tell us a lot about our target variable.
# 
# > Our target variable, **Severity** is not normally distributed.
# 
# > Our target variable is right-skewed. 
# 
# > There are multiple outliers in the variable.

# In[46]:


#skewness and kurtosis
print("Skewness: " + str(train['Severity'].skew()))
print("Kurtosis: " + str(train['Severity'].kurt()))


# ### --------> OBSERVATION
# 
# <hr>
# 
# <b>Positive Skewness</b>
# 
# These **three** charts above can tell us a lot about our target variable. There are quite a bit Skewness and Kurtosis in the target variable. 
# 
# > Our target variable, **Severity** is not normally distributed and it is right-skewed. 
# 
# > There are multiple outliers in the variable.
# 
# > Similar to our target variable distribution means the mean and median will be greater than the mode similar to this dataset. Which means more more sever case by car accident than the average case.
# 
# 
# <b>Kurtosis</b> 
# > My target variable shows an unequal level of variance across most independent variables. This **Heteroscedasticity** is a red flag for the multiple linear regression model.
# 
# *In probability theory and statistics, **Kurtosis** is the measure of the outliers present in my distribution.*
# 
# > There are many outliers in the scatter plots above that took my attention. 
# 
# 
# <hr>

# In[47]:


numeric_feats = train.dtypes[train.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# <a id="3.2"></a>
# ### 3.2 [Box Cox Transformation of (highly) skewed features](https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.special.boxcox1p.html)
# + We use the scipy function boxcox1p which computes the Box-Cox transformation of  1+x .
# 
# + Note that setting  λ=0  is equivalent to log1p used above for the target variable.
# 
# + See this page for more details on Box Cox Transformation as well as the scipy function's page

# In[48]:


# from scipy.special import boxcox1p

# # find skewness function
# def find_skewness(df):
#     numeric_feats = df.dtypes[df.dtypes != "object"].index
#     # Check the skew of all numerical features
#     skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
#     print("\nSkew in numerical features: \n")
#     skewness = pd.DataFrame({'Skew' :skewed_feats})
#     return skewness

# def log_transform(df):
#     numeric_feats = df.dtypes[df.dtypes != "object"].index
#     # Check the skew of all numerical features
#     skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
#     print("\nSkew in numerical features: \n")
#     skewness = pd.DataFrame({'Skew' :skewed_feats})
#     skewed_features = skewness[abs(skewness) > 0.75].index
#     for feat in skewed_features:
#         print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
#         df[feat] = np.log1p(df[feat])


# def boxcox_transform(df):
#     numeric_feats = df.dtypes[df.dtypes != "object"].index
#     # Check the skew of all numerical features
#     skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
#     print("\nSkew in numerical features: \n")
#     skewness = pd.DataFrame({'Skew' :skewed_feats})
#     skewed_features = skewness[abs(skewness) > 0.75].index
#     for feat in skewed_features:
#         print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
#         df[feat] = boxcox1p(df[feat], 0.15)

# boxcox_transform(train)
# boxcox_transform(test)
# boxcox_transform(validation)


# In[49]:


## Getting the correlation of all the features with target variable. 
(train.corr()**2)["Severity"].sort_values(ascending = False)[1:]


# ### ---> OBSERVATION
# 
# > These are the predictor variables sorted in a descending order starting with the most correlated one **Start_Lng**. 
# 
# > The results showed that the parameters such as Start_Lng', 'Distance(mi)', 'Humidity(%)', 'Pressure(in)', 'Wind_Speed(mph)' were the most prominent factors which increase the severity of crashes.

# <a id="3.3"></a>
# ### 3.3 Assumptions of Regression
# 
# * **Linearity ( Correct functional form )** 
# * **Homoscedasticity ( Constant Error Variance )( vs Heteroscedasticity )**
# * **Independence of Errors ( vs Autocorrelation )**
# * **Multivariate Normality ( Normality of Errors )**
# * **No or little Multicollinearity** 
# 
# > So, **How do I check regression assumptions? I fit a regression line and look for the variability of the response data along the regression line.** Let's apply this to each one of them.
# 
# > **Linearity(Correct functional form):** 
# Linear regression needs the relationship between each independent variable and the dependent variable to be linear. The linearity assumption can be tested with scatter plots. The following two examples depict two cases, where no or little linearity is present. 

# ### Removing multicollinary columns

# In[50]:


## Plot sizing. 
fig, (ax1, ax2) = plt.subplots(figsize = (12,8), ncols=2,sharey=False)
## Scatter plotting for Severity and Distance(mi).
sns.scatterplot( x = train['Distance(mi)'], y = train.Severity,  ax=ax1)
## Putting a regression line. 
sns.regplot(x=train['Distance(mi)'], y=train.Severity, ax=ax1)

## Scatter plotting for Severity and ['Wind_Speed(mph)'].
sns.scatterplot(x = train['Wind_Speed(mph)'],y = train.Severity, ax=ax2, color='pink')
## regression line for MasVnrArea and Severity.
sns.regplot(x=train['Wind_Speed(mph)'], y=train.Severity, ax=ax2);


# In[51]:


plotting_3_chart(train, 'Severity')


# Now, let's make sure that the target variable follows a normal distribution. If you want to learn more about the probability plot(Q-Q plot), try [this](https://www.youtube.com/watch?v=smJBsZ4YQZw) video. You can also check out [this](https://www.youtube.com/watch?v=9IcaQwQkE9I) one if you have some extra time.

# In[52]:


# ## transforming target variable using numpy.log1p,
# train["Severity"] = np.log1p(train["Severity"])
#
# ## Plotting the newly transformed response variable
# plotting_3_chart(train, 'Severity')


# ### ---------> Observation
# 
# <hr>
# 
# * There are multiple types of features including int, object, and float
# * No feature have missing values. 
# 
# > I want to focus on the target variable which is **Severity.** Let's create a histogram to see **if the features are normally distributed**. This is one of the assumptions of multiple linear regression. 
# 
# <hr>

# In[53]:


from scipy import stats
from scipy.stats import norm, skew #for some statistics

sns.distplot(train['Severity'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['Severity'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Severity distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['Severity'], plot=plt)
plt.show()


# As you can see, the log transformation removes the normality of errors, which solves most of the other errors we talked about above. Let's make a comparison of the pre-transformed and post-transformed state of residual plots. 

# ### ------> OBSERVATION
# > Here, we see that the pre-transformed chart on the left has heteroscedasticity, and the post-transformed chart on the right has Homoscedasticity(almost an equal amount of variance across the zero lines). It looks like a blob of data points and doesn't seem to give away any relationships. That's the sort of relationship we would like to see to avoid some of these assumptions. 

# ### Data Correlation
# 
# As we look through these scatter plots, I realized that it is time to explain the assumptions of Multiple Linear Regression. Before building a multiple linear regression model, we need to check that these assumptions below are valid.
# 
# We can already see some potentially interesting relationships between the target variable (the number of fatal accidents) and the feature variables (the remaining three columns).
# 
# To quantify the pairwise relationships that we observed in the scatter plots, we can compute the Pearson correlation coefficient matrix. The Pearson correlation coefficient is one of the most common methods to quantify correlation between variables, and by convention, the following thresholds are usually used:
# 
# 0.2 = weak
# 0.5 = medium
# 0.8 = strong
# 0.9 = very strong

# In[54]:


train.describe() 


# In[55]:


# compare severity level rate across numerical columns
pd.pivot_table(train, index = 'Severity', values = ['Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng', 'Distance(mi)', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)'])


# **-----------> OBSERVATION**
# > Diverse range of values for different features
# 
# > I might need to use scaling to unify the range of the features

# ## Multicollinearity of Features

# In[56]:


def customized_scatterplot(y, x):
        ## Sizing the plot. 
    style.use('fivethirtyeight')
    plt.subplots(figsize = (12,8))
    ## Plotting target variable with predictor variable(OverallQual)
    sns.scatterplot(y = y, x = x)

## Plot fig sizing. 
plt.style.use('ggplot')
sns.set_style('whitegrid')
plt.subplots(figsize = (30,20))
## Plotting heatmap. 

# Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(train.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(train.corr(), 
            cmap=sns.diverging_palette(20, 220, n=200), 
            mask = mask, 
            annot=True,
            fmt = ".2f",
            center = 0, 
           );

## Give title. 
plt.title("Heatmap of all the Features", fontsize = 30);


# ### ---------> OBSERVATION
# <hr>
# 
# + From the matrix I can see that the start and end GPS coordinates of the accidents are highly correlated.
# 
# + In fact, from the medium distance shown before, the end of the accident is usually close to the start, so I can consider just one of them for the machine learning models.
# 
# + Moreover, the wind chill (temperature) is directly proportional to the temperature, so we can also drop one of them.
# 
# + I can also see that the presence of a traffic signal is slightly correlated to the severity of an accident meaning that maybe traffic lights can help the traffic flow when an accident occurs.
# 
# + From the matrix I can also note that we couldn't compute the covariance with Turning_Loop, and that's because it's always False.
# 
# + Sunrise_Sunset, Nautical_Twilight, Astronomical_Twilight are redundant
# 
# <hr>

# # Drop Multicollinearity features and high p-value

# In[57]:


print("Total missing values in TRAIN:", train.isna().sum().sum())
print("Total missing values in TEST:", test.isna().sum().sum())
print("Total missing values in VALIDATION:", validation.isna().sum().sum())


# In[58]:


# select non-numeric columns
categorical = train.select_dtypes(exclude=[np.number])
categorical = categorical.columns.tolist()
print(f'List of non numeric in train dataset is {categorical}\n\n')


# select numeric columns
numeric = train.select_dtypes(include=[np.number])
numeric = numeric.columns.tolist()
print(f'List of numeric in test dataset is {numeric}\n\n')


# In[59]:


unneeded_columns = ['State', 'Side', 'City', 'County', 'Zipcode', 'Airport_Code', 'Wind_Direction', 'Weather_Condition', 'Amenity', 'Bump',
                    'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Sunrise_Sunset', 
                    'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight', 
                    'Start_Lat', 'End_Lat', 'End_Lng', 'Temperature(F)', 'Visibility(mi)', 'Precipitation(in)', 
                    'Start_Time_Month', 'Start_Time_Year', 'Start_Time_Hour', 'End_Time_Month', 'End_Time_Year', 'End_Time_Hour', 'Weather_Timestamp_Month', 'Weather_Time_Hour']

data = train.drop(unneeded_columns, axis=1)
test = test.drop(unneeded_columns, axis=1)
validation = validation.drop(unneeded_columns, axis=1)


# In[60]:


# select non-numeric columns
categorical = data.select_dtypes(exclude=[np.number])
categorical = categorical.columns.tolist()
print(f'List of non numeric in train dataset is {categorical}\n\n')


# select numeric columns
numeric = data.select_dtypes(include=[np.number])
numeric = numeric.columns.tolist()
print(f'List of numeric in test dataset is {numeric}\n\n')


# # Encoding

# ## c. Categorical Variables

# In[61]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(30,8))

for xcol, ax in zip(['Start_Lng', 'Distance(mi)', 'Humidity(%)', 'Pressure(in)', 'Wind_Speed(mph)'], axes):
    data.plot(kind='scatter', x=xcol, y='Severity', ax=ax, alpha=0.4, color='pink')


# ### -------> OBSERVATION:
# > The distribution of the scatterplot indicates that the features are somewhat correlated to the target variable

# In[62]:


# Encoding binary columns
data = data.replace([True, False], [1,0])
test = test.replace([True, False], [1,0])
validation = validation.replace([True, False], [1,0])


# Encoding nominal columns
def onehot_encode(df, columns, prefixes):
    df = df.copy()
    for column, prefix in zip(columns, prefixes):
        dummies = pd.get_dummies(df[column], prefix=prefix)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
    return df

data = onehot_encode(
    data,
    columns=['Timezone', 'Crossing', 'Traffic_Signal'],
    prefixes=['Timezone', 'Crossing', 'Traffic_Signal']
)

test = onehot_encode(
    test,
    columns=['Timezone', 'Crossing', 'Traffic_Signal'],
    prefixes=['Timezone', 'Crossing', 'Traffic_Signal']
)

validation = onehot_encode(
    validation,
    columns=['Timezone', 'Crossing', 'Traffic_Signal'],
    prefixes=['Timezone', 'Crossing', 'Traffic_Signal']
)


# In[63]:


data.head(3)


# In[64]:


formula = 'Severity ~ '+ '+'.join(data.columns[1:])
formula


# ### Ols stats model - Multivariate

# In[65]:


get_ipython().run_cell_magic('time', '', "from statsmodels.formula.api import ols\n\n# Rename the copy DataFrame to fit into stats model\ndata1 = data.rename(columns={'Distance(mi)': 'Distance_mi', 'Humidity(%)': 'Humidity_perc', 'Pressure(in)': 'Pressure_in', 'Wind_Speed(mph)': 'Wind_Speed_mph'})\n\nformula = 'Severity ~ '+ '+'.join(data1.columns[1:])\n\nmodel = ols(formula='Severity ~ Start_Lng+Distance_mi+Humidity_perc+Pressure_in+Wind_Speed_mph+Weather_Timestamp_Year+Crossing_0+Crossing_1+Traffic_Signal_0+Traffic_Signal_1', data=data1).fit()\nmodel.summary()")


# ### -------------> OBSERVATION
# 
# <hr>
# 
# ### Interpretation of the Model Coefficient, the P-value, the R-squared
# > The output above shows that, when the other variables remain constant, if we compare two applicants whose 'Weather_Timestamp_Year[T.2017]' differ by one unit, the applicant with higher 'Weather_Timestamp_Year[T.2017]' will, on average, have 0.0156 units higher 'Income'.
# > Using the P>|t| result, I can infer that the variables all independent variables are the statistically significant variables, as their p-value is less than 0.05.
# > The Adj. R-squared 0.130 indicates the amount of variability not being explained by my model that much
# 
# <hr>

# <a id="4"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>3. Feature Engineering</strong></h1>
# 
# ## Train - Test -  Validation
# 
# + Train dataset for train set
# 
# + Validation for test set
# 
# + Test for prediction on Kaggle

# In[66]:


X = data.drop(['Severity'], axis = 1)
y = data['Severity']

X_train = X
y_train = y

accident_ID = test.ID.to_list()
TEST = test.drop(['ID'], axis = 1)

X_test = validation.drop(['Severity'], axis = 1)
y_test = validation['Severity']


# # Rescale Inputs:
# Linear regression will often make more reliable predictions if you rescale input variables using standardization or normalization.

# In[67]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\n\npipeline = Pipeline([\n    ('std_scalar', StandardScaler())\n])\n\nX_train = pipeline.fit_transform(X_train)\nX_test = pipeline.transform(X_test)\nTEST = pipeline.fit_transform(TEST)")


# <a id="4"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>4. Model training</strong></h1>

# ##  Regression Evaluation Metrics
# 
# 
# Here are three common evaluation metrics for regression problems:
# 
# > - **Mean Absolute Error** (MAE) is the mean of the absolute value of the errors:
# $$\frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$
# 
# > - **Mean Squared Error** (MSE) is the mean of the squared errors:
# $$\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$
# 
# > - **Root Mean Squared Error** (RMSE) is the square root of the mean of the squared errors:
# $$\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$
# 
# > - **Residuals** (R2):
# 
# > 📌 Comparing these metrics:
# - **MAE** is the easiest to understand, because it's the average error.
# - **MSE** is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
# - **RMSE** is even more popular than MSE, because RMSE is interpretable in the "y" units.
# - **R2** is independent of each other, independent of x, normally distributed, common variance, have 0 mean
# 
# > All of these are **loss functions**, because we want to minimize them.

# In[68]:


from sklearn import metrics

def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')
    
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square


# # 📈 Linear Regression

# In[69]:


get_ipython().run_cell_magic('time', '', '\nfrom sklearn.linear_model import LinearRegression\n\nlin_reg = LinearRegression(normalize=True)\nlin_reg.fit(X_train,y_train)')


# In[70]:


# print the intercept
print(lin_reg.intercept_)


# In[71]:


# Evaluate coefficients
coeff_df = pd.DataFrame(lin_reg.coef_, X.columns, columns=['Coefficient'])
coeff_df


# > Interpreting the coefficients:
# - Holding all other features fixed, a 1 unit increase in **Distance(mi)** is associated with an **increase of 7.761550e-02**.
# - Holding all other features fixed, a 1 unit increase in **Pressure(in)** is associated with an **decrease of -8.454575e-03**.

# In[72]:


get_ipython().run_cell_magic('time', '', '# Predictions from our Model\npred = lin_reg.predict(X_test)')


# In[73]:


get_ipython().run_cell_magic('time', '', "# predictions from our model on test set\ntest_pred = lin_reg.predict(X_test)\ntrain_pred = lin_reg.predict(X_train)\n\nprint('Test set evaluation:\\n_____________________________________')\nprint_evaluate(y_test, test_pred)\nprint('Train set evaluation:\\n_____________________________________')\nprint_evaluate(y_train, train_pred)")


# In[74]:


results_df = pd.DataFrame(data=[["Linear Regression", *evaluate(y_test, test_pred)]],
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])
results_df


# In[75]:


import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

plt.figure(figsize = (20,10))

plt.plot(test_pred, label='Predicted', color='pink')
plt.plot(y_test, label='Actual', color='blue')

plt.ylabel('Severity')

plt.legend()
plt.show()


# In[76]:


get_ipython().run_cell_magic('time', '', '\ny_predict_model_lin_reg = lin_reg.predict(TEST)\n\noutput = pd.DataFrame({"ID": accident_ID, "Severity": y_predict_model_lin_reg})\n\noutput.to_csv(\'submission_lin_reg.csv\', index=False)\nprint("Submission was successfully saved!")')


# # Regularization Techniques
# 
# To overcome over-fitting, I do regularization which penalizes large coefficients. The following are the regularization algorithms.
# 
# #### Pros of Regularization
# 
# --> We can use a regularized model to reduce the dimensionality of the training dataset. Dimensionality reduction is important because of three main reasons:
# 
# --> Prevents Overfitting: A high-dimensional dataset having too many features can sometimes lead to overfitting (model captures both real and random effects).
# 
# --> Simplicity: An over-complex model having too many features can be hard to interpret especially when features are correlated with each other.
# 
# --> Computational Efficiency: A model trained on a lower dimensional dataset is computationally efficient (execution of algorithm requires less computational time).
# 
# 
# #### Cons of Regularization
# 
# --> Regularization leads to dimensionality reduction, which means the machine learning model is built using a lower dimensional dataset. This generally leads to a high bias errror.
# 
# --> If regularization is performed before training the model, a perfect balance between bias-variance tradeoff must be used.
# 

# # 📈 Ridge Regression
# 
# > Source: [scikit-learn](http://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)
# 
# > Ridge regression addresses some of the problems of **Ordinary Least Squares** by imposing a penalty on the size of coefficients. The ridge coefficients minimize a penalized residual sum of squares,
# 
# $$\min_{w}\big|\big|Xw-y\big|\big|^2_2+\alpha\big|\big|w\big|\big|^2_2$$
# 
# > $\alpha>=0$ is a complexity parameter that controls the amount of shrinkage: the larger the value of $\alpha$, the greater the amount of shrinkage and thus the coefficients become more robust to collinearity.
# 
# > Ridge regression is an L2 penalized model. Add the squared sum of the weights to the least-squares cost function.
# ***
# 
# #### Pros
# 
# --> Avoids overfitting a model.
# 
# --> The ridge estimator is preferably good at improving the least-squares estimate when there is multicollinearity.
# 
# 
# #### Cons
# 
# --> They include all the predictors in the final model.
# 
# --> They are unable to perform feature selection.
# 
# --> They shrink the coefficients towards zero.
# 
# --> They trade the variance for bias.

# In[77]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.linear_model import Ridge\n\nmodel = Ridge(alpha=100, solver='cholesky', tol=0.0001, random_state=42)\nmodel.fit(X_train, y_train)\npred = model.predict(X_test)\n\ntest_pred = model.predict(X_test)\ntrain_pred = model.predict(X_train)\n\nprint('Test set evaluation:\\n_____________________________________')\nprint_evaluate(y_test, test_pred)\nprint('====================================')\nprint('Train set evaluation:\\n_____________________________________')\nprint_evaluate(y_train, train_pred)")


# In[78]:


results_df_2 = pd.DataFrame(data=[["Ridge Regression", *evaluate(y_test, test_pred)]],
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df


# In[79]:


plt.figure(figsize = (20,10))

plt.plot(test_pred, label='Predicted', color='pink')
plt.plot(y_test, label='Actual', color='blue')

plt.ylabel('Severity')

plt.legend()
plt.show()


# In[80]:


get_ipython().run_cell_magic('time', '', '\ny_predict_Ridge = model.predict(TEST)\n\noutput = pd.DataFrame({"ID": accident_ID, "Severity": y_predict_Ridge})\n\noutput.to_csv(\'submission_Ridge.csv\', index=False)\nprint("Submission was successfully saved!")')


# # 📈 Lasso Regression
# 
# > A linear model that estimates sparse coefficients.
# 
# > Mathematically, it consists of a linear model trained with $\ell_1$ prior as regularizer. The objective function to minimize is:
# 
# $$\min_{w}\frac{1}{2n_{samples}} \big|\big|Xw - y\big|\big|_2^2 + \alpha \big|\big|w\big|\big|_1$$
# 
# > The lasso estimate thus solves the minimization of the least-squares penalty with $\alpha \big|\big|w\big|\big|_1$ added, where $\alpha$ is a constant and $\big|\big|w\big|\big|_1$ is the $\ell_1-norm$ of the parameter vector.
# ***
# 
# #### Pros
# 
# --> Avoids overfitting a model.
# 
# --> The ridge estimator is preferably good at improving the least-squares estimate when there is multicollinearity.
# 
# 
# #### Cons
# 
# --> They include all the predictors in the final model.
# 
# --> They are unable to perform feature selection.
# 
# --> They shrink the coefficients towards zero.
# 
# --> They trade the variance for bias.
# 

# In[81]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.linear_model import Lasso\n\nmodel = Lasso(alpha=0.1, \n              precompute=True, \n              positive=True, \n              selection='random',\n              random_state=42)\nmodel.fit(X_train, y_train)\n\ntest_pred = model.predict(X_test)\ntrain_pred = model.predict(X_train)\n\nprint('Test set evaluation:\\n_____________________________________')\nprint_evaluate(y_test, test_pred)\nprint('====================================')\nprint('Train set evaluation:\\n_____________________________________')\nprint_evaluate(y_train, train_pred)")


# In[82]:


results_df_2 = pd.DataFrame(data=[["Lasso Regression", *evaluate(y_test, test_pred)]],
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df


# In[83]:


plt.figure(figsize = (20,10))

plt.plot(test_pred, label='Predicted', color='pink')
plt.plot(y_test, label='Actual', color='blue')

plt.ylabel('Severity')

plt.legend()
plt.show()


# In[84]:


get_ipython().run_cell_magic('time', '', '\ny_predict_Lasso = model.predict(TEST)\n\noutput = pd.DataFrame({"ID": accident_ID, "Severity": y_predict_Lasso})\n\noutput.to_csv(\'submission_lasso.csv\', index=False)\nprint("Submission was successfully saved!")')


# # 📈 Elastic Net
# 
# > A linear regression model trained with L1 and L2 prior as regularizer. 
# 
# > This combination allows for learning a sparse model where few of the weights are non-zero like Lasso, while still maintaining the regularization properties of Ridge. 
# 
# > Elastic-net is useful when there are multiple features which are correlated with one another. Lasso is likely to pick one of these at random, while elastic-net is likely to pick both.
# 
# > A practical advantage of trading-off between Lasso and Ridge is it allows Elastic-Net to inherit some of Ridge’s stability under rotation.
# 
# > The objective function to minimize is in this case
# 
# $$\min_{w}{\frac{1}{2n_{samples}} \big|\big|X w - y\big|\big|_2 ^ 2 + \alpha \rho \big|\big|w\big|\big|_1 +
# \frac{\alpha(1-\rho)}{2} \big|\big|w\big|\big|_2 ^ 2}$$
# ***
# 
# #### Pros
# --> Doesn’t have the problem of selecting more than n predictors when n<<p, whereas LASSO saturates when n<<p.
# 
# #### Cons
# --> Computationally more expensive than LASSO or Ridge.

# In[85]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.linear_model import ElasticNet\n\nmodel = ElasticNet(alpha=0.1, l1_ratio=0.9, selection='random', random_state=42)\nmodel.fit(X_train, y_train)\n\ntest_pred = model.predict(X_test)\ntrain_pred = model.predict(X_train)\n\nprint('Test set evaluation:\\n_____________________________________')\nprint_evaluate(y_test, test_pred)\nprint('====================================')\nprint('Train set evaluation:\\n_____________________________________')\nprint_evaluate(y_train, train_pred)")


# In[86]:


results_df_2 = pd.DataFrame(data=[["Elastic Net Regression", *evaluate(y_test, test_pred)]],
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df


# In[87]:


plt.figure(figsize = (20,10))

plt.plot(test_pred, label='Predicted', color='pink')
plt.plot(y_test, label='Actual', color='blue')

plt.ylabel('Severity')

plt.legend()
plt.show()


# In[88]:


get_ipython().run_cell_magic('time', '', '\ny_predict_ElasticNet = model.predict(TEST)\n\noutput = pd.DataFrame({"ID": accident_ID, "Severity": y_predict_ElasticNet})\n\noutput.to_csv(\'submission_ElasticNet.csv\', index=False)\nprint("Submission was successfully saved!")')


# #  📈 Polynomial Regression
# > Source: [scikit-learn](http://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions)
# 
# ***
# 
# > My data might be nonlinear functions of the data. This approach maintains the generally fast performance of linear methods, while allowing them to fit a much wider range of data.
# 
# > For example, a simple linear regression can be extended by constructing polynomial features from the coefficients. In the standard linear regression case, you might have a model that looks like this for two-dimensional data:
# 
# $$\hat{y}(w, x) = w_0 + w_1 x_1 + w_2 x_2$$
# ***

# In[89]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.preprocessing import PolynomialFeatures\n\npoly_reg = PolynomialFeatures(degree=2)\n\nX_train_2_d = poly_reg.fit_transform(X_train)\nX_test_2_d = poly_reg.transform(X_test)\nTEST_2d = poly_reg.fit_transform(TEST)\n\nlin_reg = LinearRegression(normalize=True)\nlin_reg.fit(X_train_2_d,y_train)\n\ntest_pred = lin_reg.predict(X_test_2_d)\ntrain_pred = lin_reg.predict(X_train_2_d)\n\nprint('Test set evaluation:\\n_____________________________________')\nprint_evaluate(y_test, test_pred)\nprint('====================================')\nprint('Train set evaluation:\\n_____________________________________')\nprint_evaluate(y_train, train_pred)")


# In[90]:


results_df_2 = pd.DataFrame(data=[["Polynomial Regression", *evaluate(y_test, test_pred)]],
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df


# In[91]:


plt.figure(figsize = (20,10))

plt.plot(test_pred, label='Predicted', color='pink')
plt.plot(y_test, label='Actual', color='blue')

plt.ylabel('Severity')

plt.legend()
plt.show()


# In[92]:


get_ipython().run_cell_magic('time', '', '\ny_predict_poly_reg = lin_reg.predict(TEST_2d)\n\noutput = pd.DataFrame()\noutput[\'ID\'] = accident_ID\noutput[\'Severity\'] = y_predict_poly_reg\n\noutput.to_csv(\'submission_poly_reg.csv\', index=False)\nprint("Submission was successfully saved!")')


# # 📈 Stochastic Gradient Descent
# 
# > Gradient Descent is a very generic optimization algorithm capable of finding optimal solutions to a wide range of problems. The general idea of Gradient Sescent is to tweak parameters iteratively in order to minimize a cost function. Gradient Descent measures the local gradient of the error function with regards to the parameters vector, and it goes in the direction of descending gradient. Once the gradient is zero, you have reached a minimum.

# In[93]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.linear_model import SGDRegressor\n\nsgd_reg = SGDRegressor(n_iter_no_change=250, penalty=None, eta0=0.0001, max_iter=100000)\nsgd_reg.fit(X_train, y_train)\n\ntest_pred = sgd_reg.predict(X_test)\ntrain_pred = sgd_reg.predict(X_train)\n\nprint('Test set evaluation:\\n_____________________________________')\nprint_evaluate(y_test, test_pred)\nprint('====================================')\nprint('Train set evaluation:\\n_____________________________________')\nprint_evaluate(y_train, train_pred)")


# In[94]:


results_df_2 = pd.DataFrame(data=[["Stochastic Gradient Descent", *evaluate(y_test, test_pred)]],
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df


# In[95]:


import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

plt.figure(figsize = (20,10))

plt.plot(test_pred, label='Predicted', color='pink')
plt.plot(y_test, label='Actual', color='blue')

plt.ylabel('Severity')

plt.legend()
plt.show()


# In[96]:


get_ipython().run_cell_magic('time', '', '\ny_predict_poly_sgd_reg = sgd_reg.predict(TEST)\n\noutput = pd.DataFrame({"ID": accident_ID, "Severity": y_predict_poly_sgd_reg})\n\noutput.to_csv(\'submission_sgd_reg.csv\', index=False)\nprint("Submission was successfully saved!")')


# # 📈 Artficial Neural Network

# In[97]:


get_ipython().run_cell_magic('time', '', "\n# Install a pip package in the current Jupyter kernel\n!{sys.executable} -m pip install tensorflow\n\nfrom tensorflow.keras.models import Sequential\nfrom tensorflow.keras.layers import Input, Dense, Activation, Dropout\nfrom tensorflow.keras.optimizers import Adam\n\nX_train = np.array(X_train)\nX_test = np.array(X_test)\ny_train = np.array(y_train)\ny_test = np.array(y_test)\n\nmodel = Sequential()\n\nmodel.add(Dense(X_train.shape[1], activation='relu'))\nmodel.add(Dense(32, activation='relu'))\n# model.add(Dropout(0.2))\n\nmodel.add(Dense(64, activation='relu'))\n# model.add(Dropout(0.2))\n\nmodel.add(Dense(128, activation='relu'))\n# model.add(Dropout(0.2))\n\nmodel.add(Dense(512, activation='relu'))\nmodel.add(Dropout(0.1))\nmodel.add(Dense(1))\n\nmodel.compile(optimizer=Adam(0.00001), loss='mse')\n\nr = model.fit(X_train, y_train,\n              validation_data=(X_test,y_test),\n              batch_size=1,\n              epochs=11,\n              verbose=1)")


# In[98]:


get_ipython().run_cell_magic('time', '', "\ntest_pred = model.predict(X_test)\ntrain_pred = model.predict(X_train)\n\nprint('Test set evaluation:\\n_____________________________________')\nprint_evaluate(y_test, test_pred)\n\nprint('Train set evaluation:\\n_____________________________________')\nprint_evaluate(y_train, train_pred)")


# In[99]:


results_df_2 = pd.DataFrame(data=[["Artificial Neural Network", *evaluate(y_test, test_pred)]],
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df


# In[100]:


import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

plt.figure(figsize = (20,10))

plt.plot(test_pred, label='Predicted', color='pink')
plt.plot(y_test, label='Actual', color='blue')

plt.ylabel('Severity')

plt.legend()
plt.show()


# In[101]:


y_predict_model_arti = model.predict(TEST)

output = pd.DataFrame()
output['ID'] = accident_ID
output['Severity'] = y_predict_model_arti

output.to_csv('submission_arti.csv', index=False)
print("Submission was successfully saved!")


# ### -----------> OBSERVATION
# 
# <hr>
# 
# > A comparison between multiple linear regression model and the model produced by the ANN is presented in tables above ,it can be seen that the ANN model has better predictive power than the above regression models. Hoever, it requires significantly higher computing power and time
# 
# <hr>

# # 📈 Other Neural Network Models

# In[102]:


from sklearn.neural_network import MLPRegressor

# mlp_reg = MLPRegressor(activation='relu',
#                        hidden_layer_sizes=(1,),
#                        solver='lbfgs',
#                        verbose=True,
#                        max_iter=100,
#                        random_state=42)
#
# # RMSE: 0.576582

# mlp_reg = MLPRegressor(activation = 'relu',
#                        hidden_layer_sizes= (5, ),
#                        solver='lbfgs',
#                        verbose=True,
#                        max_iter=100)
#
# # RMSE: 0.571221

# mlp_reg = MLPRegressor(activation = 'relu',
#                        hidden_layer_sizes= (2, 4),
#                        solver='lbfgs',
#                        verbose=True,
#                        max_iter=100)
#
# # RMSE: 0.570725

# mlp_reg = MLPRegressor(activation = 'logistic',
#                        hidden_layer_sizes= (2, 4),
#                        solver='lbfgs',
#                        verbose=True,
#                        max_iter=1000,
#                        learning_rate_init=0.001,
#                        learning_rate='adaptive',
#                        early_stopping=True,
#                        validation_fraction=0.1,
#                        n_iter_no_change=10)
# # RMSE: 0.561204

mlp_reg = MLPRegressor(activation = 'relu',
                       hidden_layer_sizes= (2, 6, 4),
                       solver='lbfgs',
                       verbose=True,
                       max_iter=1000,
                       learning_rate_init=0.001)

# 0.556085

mlp_reg.fit(X_train, y_train)
y_pred = mlp_reg.predict(X_test)
mlp_reg.score(X_train, y_train)
r2_score(y_test, y_pred)


# In[103]:


results_df_2 = pd.DataFrame(data=[["Other Neural Network", *evaluate(y_test, test_pred)]],
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df


# In[104]:


get_ipython().run_cell_magic('time', '', 'y_predict_mlp_reg = mlp_reg.predict(TEST)\n\noutput = pd.DataFrame({"ID": accident_ID, "Severity": y_predict_mlp_reg})\n\noutput.to_csv(\'submission_mlp_reg.csv\', index=False)\nprint("Submission was successfully saved!")')


# ### -----------> OBSERVATION
# 
# <hr>
# 
# > Simpler models worked well for my data and having a single layer with many neurons did not perform well. ReLU activation worked better than logistic activation.
# 
# <hr>

# # 📈 Random Forest Regressor

# In[105]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.ensemble import RandomForestRegressor\n\nrf_reg = RandomForestRegressor(n_estimators=1000)\nrf_reg.fit(X_train, y_train)\n\ntest_pred = rf_reg.predict(X_test)\ntrain_pred = rf_reg.predict(X_train)\n\nprint('Test set evaluation:\\n_____________________________________')\nprint_evaluate(y_test, test_pred)\n\nprint('Train set evaluation:\\n_____________________________________')\nprint_evaluate(y_train, train_pred)")


# In[106]:


results_df_2 = pd.DataFrame(data=[["Random Forest Regressor", *evaluate(y_test, test_pred), 0]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df


# In[107]:


import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

plt.figure(figsize = (20,10))

plt.plot(test_pred, label='Predicted', color='pink')
plt.plot(y_test, label='Actual', color='blue')

plt.ylabel('Severity')

plt.legend()
plt.show()


# In[108]:


y_predict_rf_reg = rf_reg.predict(TEST)

output = pd.DataFrame()
output['ID'] = accident_ID
output['Severity'] = y_predict_rf_reg

output.to_csv('submission_rf_reg.csv', index=False)
print("Submission was successfully saved!")


# <a id="5"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>5. Model Comparision and Export</strong></h1>

# In[109]:


results_df.set_index('Model', inplace=True)
results_df['R2 Square'].plot(kind='barh', figsize=(12, 8), color='pink')


# In[111]:


# import joblib

# # Save the model as a pickle in a file
# joblib.dump(lin_reg, 'lin_reg.pkl')
# joblib.dump(model, 'arti.pkl')
# joblib.dump(rf_reg, 'RandomForest.pkl')


# ### -----------> OBSERVATION
# > The Random Forest Regressor model has the best performance in terms of R2 Square. The Random Forest Regressor model is chosen as the superior approach in predicting the number and severity of crashes. Besides, performance and sensitivity analysis prove the excellent performance and validation.
# 
# > List of feature selected: ['Timezone', 'Crossing', 'Traffic_Signal', 'Weather_Timestamp_Year', 'Severity', 'Start_Lng', 'Distance(mi)', 'Humidity(%)', 'Pressure(in)', 'Wind_Speed(mph)']

# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>5. Summary</strong></h1>
# 
# > The maximum number of accidents have taken place at around 4-5P.M, and a relatively high has taken place from 7–8 AM. This can result from the rush hour that most people come to and back to work, school, and home.
# 
# > The state has the most number of accidents in California (28%), followed by Florida(10%) in the USA (2016-2020).
# 
# > The density of points is more at the eastern and western coasts than in the middle of the country, indicating that more accidents were recorded at the two sides from February 2016 to Dec 2020 in the Contiguous United States rather than its middle part.
# 
# > The graph shows the accident is more or less proportional to the severity
# 
# > The accidents with a severity level of 4 have the longest distance.
# The longer the distance, the more severe the accidents
# 
# > Fatal accidents occurred near a traffic signal, junction and crossing were present. The driver might fail to pay attention before pulling out due to impatience, impairment of one form or another, or a simple failure to judge the distance and speed of an oncoming vehicle.
# 
# > The most common weather condition is Fair, followed by Mostly Cloudy and Overcast.
# 
# > Surprisingly, significantly higher accident cases were recorded from September to November, but the more severe impact on traffic happens around February to May.
# 
# > The best model I get overall is Random Forest Regressor  with the lowest RMSE value on the test dataset

# <a id="7"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>6. References</strong></h1>
# 
# > Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. [“A Countrywide Traffic Accident Dataset.”](https://arxiv.org/abs/1906.05409), arXiv preprint arXiv:1906.05409 (2019). Access Nov 27, 2021.
# 
# > Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath. [“Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights.”](https://arxiv.org/abs/1909.09638) In proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, ACM, 2019. Access Nov 27, 2021.
# 
# > SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM. Access Dec 2, 2021.
# 
# ### ANN model:
# 
# > M. Çodur and A. Tortum, "An Artificial Neural Network Model for Highway Accident Prediction: A Case Study of Erzurum, Turkey", PROMET - Traffic&Transportation, vol. 27, no. 3, pp. 217-225, 2015. Available: 10.7307/ptt.v27i3.1551 [Accessed 18 December 2021].
# 
# > A. Alqatawna, A. Rivas Álvarez and S. García-Moreno, "Comparison of Multivariate Regression Models and Artificial Neural Networks for Prediction Highway Traffic Accidents in Spain: A Case Study", Transportation Research Procedia, vol. 58, pp. 277-284, 2021. Available: 10.1016/j.trpro.2021.11.038 [Accessed 18 December 2021].
# 
# >M. Ghasedi, M. Sarfjoo and I. Bargegol, "Prediction and Analysis of the Severity and Number of Suburban Accidents Using Logit Model, Factor Analysis and Machine Learning: A case study in a developing country", SN Applied Sciences, vol. 3, no. 1, 2021. Available: 10.1007/s42452-020-04081-3 [Accessed 19 December 2021].

# <a id="7"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>7. Appendix</strong></h1>
# 
# **Link to the github repo:** https://github.com/tnathu-ai/severity_prediction_linear_regression
