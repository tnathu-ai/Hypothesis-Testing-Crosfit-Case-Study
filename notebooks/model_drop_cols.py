#!/usr/bin/env python
# coding: utf-8

# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Assignment 2: Data Modelling</strong></h1>
# 

# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Table of Content</strong></h1>
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

# In[ ]:


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


# In[2]:


import sklearn

# check the version of the package
print(sklearn.__version__)
print(np.__version__)
print(pd.__version__)


# In[3]:


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

# In[4]:


# drop unique id
df = df.drop('athlete_id',axis=1)
# drop multicolinerality
# Drop non-needed columns
df = df.drop(['retrieved_datetime_x', 'retrieved_datetime_y', 'year', 'stage', 'scaled', 'howlong'], axis=1)
# select numeric columns
df = df.select_dtypes(include=[np.number])
df.sample(3)


# In[5]:


get_ipython().system('pip install fast_ml')
from fast_ml.model_development import train_valid_test_split

X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(df, target = 'score',
                                                                            train_size=0.8, valid_size=0.1, test_size=0.1)

print(X_train.shape), print(y_train.shape)
print(X_valid.shape), print(y_valid.shape)
print(X_test.shape), print(y_test.shape)


# In[6]:


print("Total missing values in TRAIN:", X_train.isna().sum().sum())
print("Total missing values in TEST:", X_test.isna().sum().sum())
print("Total missing values in VALIDATION:", X_valid.isna().sum().sum())


# In[7]:


print (f"Train has {X_train.shape[0]} rows and {X_train.shape[1]} columns")
print (f"Test has {X_test.shape[0]} rows and {X_test.shape[1]} columns")
print (f"Validation has {X_valid.shape[0]} rows and {X_valid.shape[1]} columns")


# In[8]:


# gives us statistical info about the numerical variables. 
X_train.describe().T


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

# In[10]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\n\npipeline = Pipeline([\n    ('std_scalar', StandardScaler())\n])\n\nX_train = pipeline.fit_transform(X_train)\nX_test = pipeline.transform(X_test)")


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

# In[11]:


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


# In[12]:


get_ipython().run_cell_magic('time', '', '\nfrom sklearn.linear_model import LinearRegression\n\nlin_reg = LinearRegression(normalize=True)\nlin_reg.fit(X_train,y_train)')


# In[13]:


get_ipython().run_cell_magic('time', '', '\nfrom sklearn.linear_model import LinearRegression\n\nlin_reg = LinearRegression(normalize=True)\nlin_reg.fit(X_train,y_train)')


# > Interpreting the coefficients:
# - Holding all other features fixed, a 1 unit increase in **Distance(mi)** is associated with an **increase of 7.761550e-02**.
# - Holding all other features fixed, a 1 unit increase in **Pressure(in)** is associated with an **decrease of -8.454575e-03**.

# In[14]:


get_ipython().run_cell_magic('time', '', "# predictions from our model on test set\ntest_pred = lin_reg.predict(X_test)\ntrain_pred = lin_reg.predict(X_train)\n\nprint('Test set evaluation:\\n_____________________________________')\nprint_evaluate(y_test, test_pred)\nprint('Train set evaluation:\\n_____________________________________')\nprint_evaluate(y_train, train_pred)")


# In[15]:


get_ipython().run_cell_magic('time', '', "# predictions from our model on test set\ntest_pred = lin_reg.predict(X_test)\ntrain_pred = lin_reg.predict(X_train)\n\nprint('Test set evaluation:\\n_____________________________________')\nprint_evaluate(y_test, test_pred)\nprint('Train set evaluation:\\n_____________________________________')\nprint_evaluate(y_train, train_pred)")


# In[16]:


results_df = pd.DataFrame(data=[["Linear Regression", *evaluate(y_test, test_pred)]],
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])
results_df


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

# In[17]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.linear_model import Ridge\n\nmodel = Ridge(alpha=100, solver='cholesky', tol=0.0001, random_state=42)\nmodel.fit(X_train, y_train)\npred = model.predict(X_test)\n\ntest_pred = model.predict(X_test)\ntrain_pred = model.predict(X_train)\n\nprint('Test set evaluation:\\n_____________________________________')\nprint_evaluate(y_test, test_pred)\nprint('====================================')\nprint('Train set evaluation:\\n_____________________________________')\nprint_evaluate(y_train, train_pred)")


# In[18]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.linear_model import Ridge\n\nmodel = Ridge(alpha=100, solver='cholesky', tol=0.0001, random_state=42)\nmodel.fit(X_train, y_train)\npred = model.predict(X_test)\n\ntest_pred = model.predict(X_test)\ntrain_pred = model.predict(X_train)\n\nprint('Test set evaluation:\\n_____________________________________')\nprint_evaluate(y_test, test_pred)\nprint('====================================')\nprint('Train set evaluation:\\n_____________________________________')\nprint_evaluate(y_train, train_pred)")


# In[19]:


results_df_2 = pd.DataFrame(data=[["Ridge Regression", *evaluate(y_test, test_pred)]],
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df


# In[20]:


# %%time
#
# y_predict_Ridge = model.predict(TEST)
#
# output = pd.DataFrame({"ID": accident_ID, "score": y_predict_Ridge})
#
# output.to_csv('submission_Ridge.csv', index=False)
# print("Submission was successfully saved!")


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

# In[21]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.linear_model import Lasso\n\nmodel = Lasso(alpha=0.1, \n              precompute=True, \n              positive=True, \n              selection='random',\n              random_state=42)\nmodel.fit(X_train, y_train)\n\ntest_pred = model.predict(X_test)\ntrain_pred = model.predict(X_train)\n\nprint('Test set evaluation:\\n_____________________________________')\nprint_evaluate(y_test, test_pred)\nprint('====================================')\nprint('Train set evaluation:\\n_____________________________________')\nprint_evaluate(y_train, train_pred)")


# In[22]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.linear_model import Lasso\n\nmodel = Lasso(alpha=0.1, \n              precompute=True, \n              positive=True, \n              selection='random',\n              random_state=42)\nmodel.fit(X_train, y_train)\n\ntest_pred = model.predict(X_test)\ntrain_pred = model.predict(X_train)\n\nprint('Test set evaluation:\\n_____________________________________')\nprint_evaluate(y_test, test_pred)\nprint('====================================')\nprint('Train set evaluation:\\n_____________________________________')\nprint_evaluate(y_train, train_pred)")


# In[23]:


results_df_2 = pd.DataFrame(data=[["Lasso Regression", *evaluate(y_test, test_pred)]],
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df


# In[24]:


plt.figure(figsize = (20,10))

plt.plot(test_pred, label='Predicted', color='pink')
plt.plot(y_test, label='Actual', color='blue')

plt.ylabel('score')

plt.legend()
plt.show()


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

# In[25]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.linear_model import ElasticNet\n\nmodel = ElasticNet(alpha=0.1, l1_ratio=0.9, selection='random', random_state=42)\nmodel.fit(X_train, y_train)\n\ntest_pred = model.predict(X_test)\ntrain_pred = model.predict(X_train)\n\nprint('Test set evaluation:\\n_____________________________________')\nprint_evaluate(y_test, test_pred)\nprint('====================================')\nprint('Train set evaluation:\\n_____________________________________')\nprint_evaluate(y_train, train_pred)")


# In[26]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.linear_model import ElasticNet\n\nmodel = ElasticNet(alpha=0.1, l1_ratio=0.9, selection='random', random_state=42)\nmodel.fit(X_train, y_train)\n\ntest_pred = model.predict(X_test)\ntrain_pred = model.predict(X_train)\n\nprint('Test set evaluation:\\n_____________________________________')\nprint_evaluate(y_test, test_pred)\nprint('====================================')\nprint('Train set evaluation:\\n_____________________________________')\nprint_evaluate(y_train, train_pred)")


# In[27]:


results_df_2 = pd.DataFrame(data=[["Elastic Net Regression", *evaluate(y_test, test_pred)]],
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df


# In[28]:


plt.figure(figsize = (20,10))

plt.plot(test_pred, label='Predicted', color='pink')
plt.plot(y_test, label='Actual', color='blue')

plt.ylabel('score')

plt.legend()
plt.show()


# In[29]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.preprocessing import PolynomialFeatures\n\npoly_reg = PolynomialFeatures(degree=2)\n\nX_train_2_d = poly_reg.fit_transform(X_train)\nX_test_2_d = poly_reg.transform(X_test)\n\nlin_reg = LinearRegression(normalize=True)\nlin_reg.fit(X_train_2_d,y_train)\n\ntest_pred = lin_reg.predict(X_test_2_d)\ntrain_pred = lin_reg.predict(X_train_2_d)\n\nprint('Test set evaluation:\\n_____________________________________')\nprint_evaluate(y_test, test_pred)\nprint('====================================')\nprint('Train set evaluation:\\n_____________________________________')\nprint_evaluate(y_train, train_pred)")


# In[30]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.preprocessing import PolynomialFeatures\n\npoly_reg = PolynomialFeatures(degree=2)\n\nX_train_2_d = poly_reg.fit_transform(X_train)\nX_test_2_d = poly_reg.transform(X_test)\n\nlin_reg = LinearRegression(normalize=True)\nlin_reg.fit(X_train_2_d,y_train)\n\ntest_pred = lin_reg.predict(X_test_2_d)\ntrain_pred = lin_reg.predict(X_train_2_d)\n\nprint('Test set evaluation:\\n_____________________________________')\nprint_evaluate(y_test, test_pred)\nprint('====================================')\nprint('Train set evaluation:\\n_____________________________________')\nprint_evaluate(y_train, train_pred)")


# In[31]:


results_df_2 = pd.DataFrame(data=[["Polynomial Regression", *evaluate(y_test, test_pred)]],
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df


# In[32]:


plt.figure(figsize = (20,10))

plt.plot(test_pred, label='Predicted', color='pink')
plt.plot(y_test, label='Actual', color='blue')

plt.ylabel('score')

plt.legend()
plt.show()


# In[33]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.linear_model import SGDRegressor\n\nsgd_reg = SGDRegressor(n_iter_no_change=250, penalty=None, eta0=0.0001, max_iter=100000)\nsgd_reg.fit(X_train, y_train)\n\ntest_pred = sgd_reg.predict(X_test)\ntrain_pred = sgd_reg.predict(X_train)\n\nprint('Test set evaluation:\\n_____________________________________')\nprint_evaluate(y_test, test_pred)\nprint('====================================')\nprint('Train set evaluation:\\n_____________________________________')\nprint_evaluate(y_train, train_pred)")


# In[34]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.linear_model import SGDRegressor\n\nsgd_reg = SGDRegressor(n_iter_no_change=250, penalty=None, eta0=0.0001, max_iter=100000)\nsgd_reg.fit(X_train, y_train)\n\ntest_pred = sgd_reg.predict(X_test)\ntrain_pred = sgd_reg.predict(X_train)\n\nprint('Test set evaluation:\\n_____________________________________')\nprint_evaluate(y_test, test_pred)\nprint('====================================')\nprint('Train set evaluation:\\n_____________________________________')\nprint_evaluate(y_train, train_pred)")


# In[35]:


results_df_2 = pd.DataFrame(data=[["Stochastic Gradient Descent", *evaluate(y_test, test_pred)]],
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df


# In[36]:


import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

plt.figure(figsize = (20,10))

plt.plot(test_pred, label='Predicted', color='pink')
plt.plot(y_test, label='Actual', color='blue')

plt.ylabel('score')

plt.legend()
plt.show()


# In[ ]:





# In[37]:


# from sklearn.neural_network import MLPRegressor

# mlp_reg = MLPRegressor(activation = 'relu',
#                        hidden_layer_sizes= (2, 6, 4),
#                        solver='lbfgs',
#                        verbose=True,
#                        max_iter=1000,
#                        learning_rate_init=0.001)

a

# mlp_reg.fit(X_train, y_train)
# y_pred = mlp_reg.predict(X_test)
# mlp_reg.score(X_train, y_train)
# r2_score(y_test, y_pred)


# ### -----------> OBSERVATION
# 
# <hr>
# 
# > Simpler models worked well for my data and having a single layer with many neurons did not perform well. ReLU activation worked better than logistic activation.
# 
# <hr>

# # Random Forest Regressor

# In[40]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.ensemble import RandomForestRegressor\n\nrf_reg = RandomForestRegressor(n_estimators=1000)\nrf_reg.fit(X_train, y_train)\n\ntest_pred = rf_reg.predict(X_test)\ntrain_pred = rf_reg.predict(X_train)\n\nprint('Test set evaluation:\\n_____________________________________')\nprint_evaluate(y_test, test_pred)\n\nprint('Train set evaluation:\\n_____________________________________')\nprint_evaluate(y_train, train_pred)")


# In[41]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.ensemble import RandomForestRegressor\n\nrf_reg = RandomForestRegressor(n_estimators=1000)\nrf_reg.fit(X_train, y_train)\n\ntest_pred = rf_reg.predict(X_test)\ntrain_pred = rf_reg.predict(X_train)\n\nprint('Test set evaluation:\\n_____________________________________')\nprint_evaluate(y_test, test_pred)\n\nprint('Train set evaluation:\\n_____________________________________')\nprint_evaluate(y_train, train_pred)")


# In[45]:


# results_df_2 = pd.DataFrame(data=[["Random Forest Regressor", *evaluate(y_test, test_pred), 0]], 
#                             columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])
# results_df = results_df.append(results_df_2, ignore_index=True)
# results_df


# In[43]:


import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

plt.figure(figsize = (20,10))

plt.plot(test_pred, label='Predicted', color='pink')
plt.plot(y_test, label='Actual', color='blue')

plt.ylabel('score')

plt.legend()
plt.show()


# <a id="5"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>5. Model Comparision and Export</strong></h1>

# In[48]:


# results_df.set_index('Model', inplace=True)
# results_df['R2 Square'].plot(kind='barh', figsize=(12, 8), color='pink')


# In[ ]:


# import joblib

# # Save the model as a pickle in a file
# joblib.dump(lin_reg, 'lin_reg.pkl')
# joblib.dump(model, 'arti.pkl')
# joblib.dump(rf_reg, 'RandomForest.pkl')


# <a id="7"></a>
# <h1 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>6. References</strong></h1>
# 
# > Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. [“A Countrywide Traffic Accident Dataset.”](https://arxiv.org/abs/1906.05409), arXiv preprint arXiv:1906.05409 (2019). Access Nov 27, 2021.
# 
# > Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath. [“Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights.”](https://arxiv.org/abs/1909.09638) In proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, ACM, 2019. Access Nov 27, 2021.
# 
# > SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM. Access Dec 2, 2021.

# In[ ]:




