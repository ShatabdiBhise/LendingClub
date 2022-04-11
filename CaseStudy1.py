#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


dataset = pd.read_csv('loans_full_schema.csv')
dataset.head()


# ## I) Descibe the dataset

# In[3]:


len(dataset)


# In[4]:


dataset.isnull().any()


# In[5]:


dataset.dtypes


# ## Description
# There are total 10,000 entries in the dataset with 55 variables or features. There are 10 columns within this dataset which contain a null value. Some of the null values make sense such as annual_income_join or verification_income_join because this indicates that the person might have made an individual loan. The dataset primarily has 3 data-types i.e integer, float and objects. 
# 
# The dataset also has a combination of categorical (both ordinal and nominal) as well as numerical (continuous and discrete) data. Some examples of ordinal - grade, sub_grade; nominal - homeownership, state; continuous - interest rate, annual_income; discrete - months_since_90d_late).
# 
# ## Issues
# The dataset has a lot of features that have null values in them and even though some of them make sense, some of the feature values could have been divised in such a way that they wouldn't have been null. For example, 'months_since_90d_late' could have the value of 0 for the individuals who are not late since it's been 0 months since they have been 90 days late or it could have been set to -1 to indicate that they are infact not late. 
# 
# The dataset could also have been divided into two datasets based on whether the application was for an individual role or joint as a single dataset gives rise to redundant features such as 'annual_income', 'annual_income_joint'. Also 10,000 entries seem less for a dataset which has 55 features. 

# ## II) Data Visualizations

# ### a) Number of Individuals per loan_status

# In[6]:


np.unique(dataset['loan_status'])


# In[7]:


loan_stat = ['Charged Off', 'Current', 'Fully Paid', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
count = []
for i in loan_stat:
    count.append(len(dataset[dataset['loan_status'] == i]))


# In[8]:


fig = plt.figure(figsize = (10, 5))
 
plt.barh(loan_stat, count, color ='orange')
 
plt.xlabel("No. of Individuals")
plt.ylabel("Loan Status")
plt.title("Number of individuals per loan status")

for index, value in enumerate(count):
    plt.text(value, index, str(value))
plt.show()


# ##### Observation
# Out of the total 10,000 individuals, it seems that the maximum number of individuals are either currently paying off the loan (9375 individuals) or have already paid off the loan (447 individuals). Less than 200 individuals are late in paying off their loans

# ### b) Top 10 job titles with maximum number of credit inquiries (Which job titles had the most credit inquiries?)

# In[9]:


frequent_job_title = dataset['emp_title'].value_counts().index.tolist()[:10]
frequent_job_title


# In[10]:


count2 = []
for i in frequent_job_title:
    df = dataset[dataset['emp_title']==i]
    c = len(df[df['months_since_last_credit_inquiry']!=0])
    count2.append(int(c))
    
count2


# In[11]:


df2 = pd.DataFrame()
df2['Job_title'] = frequent_job_title
df2['No_of_individuals_inquired'] = count2
df2 = df2.sort_values('No_of_individuals_inquired', ascending=False)
df2.head()


# In[12]:


fig = plt.figure(figsize = (14, 6))
 
plt.bar(df2['Job_title'], df2['No_of_individuals_inquired'], color =[0,0.4770, 0.7910])
 
plt.xlabel('Job Title')
plt.ylabel("Number of credit inquiries")
plt.title("Top 10 professions who took a loan arranged by the number of credit inquiries")

for index, value in enumerate(df2['No_of_individuals_inquired']):
    plt.text(index, value, str(value))
plt.show()


# #### Observation
# The graph shows the first top 10 job titles who took a loan and then among these top 10 job titles, it ranks the titles according to the number of credit inquiries for each profession. 
# 
# We can see from the plot that the managers were inquired the most, followed by teachers and owners. This might also be an indication that among all the job types, managers take the most number of loans as the individuals' credits are inquired only when they have applied for a loan. 

# ### c) Number of applicants per loan type

# In[13]:


loan_type = np.unique(dataset['application_type'])
loan_type


# In[14]:


ind = []
joint = []

for i in frequent_job_title:
    df = dataset[dataset['emp_title']==i]
    ind.append(len(df[df['application_type'] == 'individual']))
    joint.append(len(df[df['application_type'] == 'joint']))
    


# In[15]:


fig = plt.figure(figsize = (14, 6))

X_axis = np.arange(len(frequent_job_title))
  
plt.bar(X_axis - 0.2, ind, 0.4, label = 'Individual Application', color='orange')
plt.bar(X_axis + 0.2, joint, 0.4, label = 'Joint Application', color='grey')
  
plt.xticks(X_axis, frequent_job_title)
plt.xlabel("Job Title")
plt.ylabel("Number of Applications")
plt.title("Number of individual and joint applications per top 10 job titles")
plt.legend()

for index, value in enumerate(ind):
    plt.text(index, value, str(value))
    
for index, value in enumerate(joint):
    plt.text(index, value, str(value))
    
plt.show()


# #### Observation
# 
# The people seem to prefer taking an idividual loan application over joint loan application irrespective of the job title.
# 
# The plot shows the distribution of individuals based on their job type and loan application type. This plot is also an indication that the managers take the most amount of loans

# ### d) Percentage of individuals belonging to different home ownership types

# In[16]:


home = np.unique(dataset['homeownership'])
home


# In[17]:


count = []

for i in home:
    count.append(len(dataset[dataset['homeownership'] == i]))
    
count


# In[18]:


import matplotlib.pyplot as plt

cols = ['g','c','r']

plt.pie(count, labels=home, colors=cols, startangle=90, shadow= True, autopct='%1.1f%%')

plt.title('Percentage of Indivduals based on their home ownership')

plt.show()


# #### Observation
# The pie chart gives the distribution of the indivduals based on the ownership of their homes. Among all the individuals taking a loan, about 50% of the individuals already seem to have a mortgage on their homes.

# ### e) Analyizing the loan amount and interest_rate

# In[19]:


fig = plt.figure(figsize =(8, 5))
 
plt.subplot(1, 2, 1) 
plt.boxplot(dataset['loan_amount'])
plt.title('Loan Amount')
plt.ylabel('Amount')

plt.subplot(1, 2, 2)
plt.boxplot(dataset['interest_rate'])
plt.title('Interest Rate')
plt.ylabel('Rate')
 
plt.show()


# #### Observation
# 
# This is boxplot for analyzing the loan amount and interest rate. The loan amount differs from a little over 0 to 40,000 with the median loan amount being 15,000. But we can say that the maximum anounts of loan are in the range of about 7500 to 25,000. There are no outliers in this distribution, that is no exceptional cases. The distribution is also right skewed, that means majority of the data is located in the lower range of the amount. The distribution is also more spread than the interest_rate, that is there is more variation.
# 
# The interest rate ranges between 5 to about 23 for maximum number of individuals and primarily lies between 5 and 15. There are multiple outliers in this distribution, that is more than 20 30 individuals who have an interest rate higher than 23, some up till more than 30 interest rate. The median value looks like about 12. The distribution is also more compact and is right skewed, that means more individuals lie on the lower side of the spectrum

# ## III) Predicting the interest rate
# 
# I experimented on various models for predicting the interest rate for an individual. The models tried on were Linear Regression, Lasso Regression, Decision Tree, Random Forest, kNN, SVM, and Neural Nets. (Neural Nets is in the google colab link as my Jupyter was not supporting keras). 
# 
# Among all the models studies, Decision Tree, Random Forest and Neural Nets seem to work the best.
# 
# Preprocessing - There were many columns with the Nan values. I had to first convert these values to 0. There was also categorical data present in the dataset which had to be encoded into a numerical value so that the dataset could work on maximum numbers of algorithms without any problem. Feature selection was then done to select the best features that would enhance the model. Root Mean Squared Error is the evaluation metric used.
# 
# The approach starts with the basic model with a high RMSE and ends with the Random Forest model which works best for the dataset

# In[20]:


from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


# In[21]:


dataset = pd.read_csv('loans_full_schema.csv')
dataset.head()


# In[22]:


#Some features were initially dropped that I believed would not affect the interest rate much

X = dataset.drop(['interest_rate', 'state', 'emp_title', 'loan_purpose','paid_total', 'paid_principal', 
                 'paid_interest', 'paid_late_fees', 'term', 'installment','issue_month'], axis=1)
Y = dataset['interest_rate']
X


# In[23]:


X.dtypes


# In[24]:


# Categorical variables are encoded to numerical values

X['homeownership'] = pd.factorize(dataset['homeownership'])[0]
X['verified_income'] = pd.factorize(dataset['verified_income'])[0]
X['verification_income_joint'] = pd.factorize(dataset['verification_income_joint'])[0]
X['application_type'] = pd.factorize(dataset['application_type'])[0]
X['grade'] = pd.factorize(dataset['grade'])[0]
X['sub_grade'] = pd.factorize(dataset['sub_grade'])[0]
X['loan_status'] = pd.factorize(dataset['loan_status'])[0]
X['initial_listing_status'] = pd.factorize(dataset['initial_listing_status'])[0]
X['disbursement_method'] = pd.factorize(dataset['disbursement_method'])[0]

X.head()


# In[26]:


#Removing NaNs
X.fillna(0, inplace=True)


# In[27]:


Y.reset_index(drop=True)


# In[28]:


#Using SelectKBest model to select the top features

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

test = SelectKBest(score_func=f_classif, k=4)
fit = test.fit(X, Y)

pd.set_option('precision', 3)
print(fit.scores_)


# In[30]:


# Selecting all the features who have the fit score more than 3 (either in the positive direction or negative)

features = []

for i in range(len(fit.scores_)):
    if fit.scores_[i] > 3.0:
        col = X.columns[i]
        features.append(col)


# In[31]:


# top features
features


# In[33]:


# new dataframe with the selected features
new_X = X[features]
new_X


# In[34]:


# Normalizing
X2 = StandardScaler().fit_transform(X)


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(X2, Y, test_size = 0.30, shuffle=True)


# ### Linear Regression

# In[44]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


# In[45]:


modelLR = LinearRegression()

modelLR.fit(X_train, y_train)

y_pred = modelLR.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred, squared=True))

print("Root Mean Squared Error: ", rmse)


# ### Lasso Regression

# In[46]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from numpy import absolute
from numpy import mean
from numpy import std


# In[47]:


X_train, X_test, y_train, y_test = train_test_split(X2, Y, test_size = 0.30, shuffle=True)


# In[49]:


modelL = Lasso(alpha=0.5)

modelL.fit(X_train, y_train)

y_pred = modelL.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred, squared=True))

print("Root Mean Squared Error: ", rmse)


# In[48]:


#Tried Lasso Regression with cross validation

modelLasso = Lasso(alpha=0.005)

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

scores = cross_val_score(modelLasso, X_train, y_train, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)

scores = absolute(scores)

print('Mean RMSE: %.3f (%.3f)' % (mean(scores), std(scores)))


# ### Decision Tree

# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X2, Y, test_size = 0.30, shuffle=True)


# In[52]:


from sklearn.tree import DecisionTreeRegressor 
  
regressor = DecisionTreeRegressor(random_state = 0) 
  
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred, squared=True))

print("Root Mean Squared Error: ", rmse)


# ### Random Forest

# In[53]:


from sklearn.ensemble import RandomForestRegressor


# In[54]:


X_train, X_test, y_train, y_test = train_test_split(X2, Y, test_size = 0.30, shuffle=True)


# In[55]:


regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

regressor.fit(X_train, y_train)

Y_pred = regressor.predict(X_test)  # test the output by changing values

rmse = np.sqrt(mean_squared_error(y_test, Y_pred, squared=True))

print("Root Mean Squared Error: ", rmse)


# ## Imp:
# Neural Network Implementation is in the Google Colab Notebook.

# ### Enhancements
# 
# During the preprocessing step, I would try to find a better way to replace the NaN values. I tried 2 different feature selection methods (Extra Tree Regressor and SelectKBest), I would have loved to try other methods for feature selection such as selectFromModel. I would have also fine-tuned my neural network model to give better performance, added, subtracted some layers, tried different optimizers and losses etc. 

# In[ ]:




