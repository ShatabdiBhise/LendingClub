# -*- coding: utf-8 -*-
"""CaseStudy1_NN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lBwIV7tn8Ba0BsUXXeK7EKVj1VlEf3hn
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

dataset = pd.read_csv('loans_full_schema.csv')
dataset.head()

X = dataset.drop(['interest_rate', 'state', 'emp_title', 'loan_purpose','paid_total', 'paid_principal', 
                 'paid_interest', 'paid_late_fees', 'term', 'installment','issue_month'], axis=1)
Y = dataset['interest_rate']
X

X['homeownership'] = pd.factorize(dataset['homeownership'])[0]
X['verified_income'] = pd.factorize(dataset['verified_income'])[0]
X['verification_income_joint'] = pd.factorize(dataset['verification_income_joint'])[0]
X['application_type'] = pd.factorize(dataset['application_type'])[0]
X['grade'] = pd.factorize(dataset['grade'])[0]
X['sub_grade'] = pd.factorize(dataset['sub_grade'])[0]
X['loan_status'] = pd.factorize(dataset['loan_status'])[0]
X['initial_listing_status'] = pd.factorize(dataset['initial_listing_status'])[0]
X['disbursement_method'] = pd.factorize(dataset['disbursement_method'])[0]

X

X.fillna(0, inplace=True)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

test = SelectKBest(score_func=f_classif, k=4)
fit = test.fit(X, Y)

pd.set_option('precision', 3)
print(fit.scores_)

features = []

for i in range(len(fit.scores_)):
    if fit.scores_[i] > 3.0 or fit.scores_[i]<-3.0:
        col = X.columns[i]
        features.append(col)

features

new_X = X[features]
new_X

y1 = np.array(Y)

verifiedIncome = new_X.verified_income.astype("category").cat.codes
verifiedIncomecat = pd.Series(verifiedIncome)

gradecat=new_X.grade.astype("category").cat.codes
gradecat=pd.Series(gradecat)

subgradecat=new_X.sub_grade.astype("category").cat.codes
subgradecat=pd.Series(subgradecat)

loanStatuscat=new_X.loan_status.astype("category").cat.codes
loanStatuscat=pd.Series(loanStatuscat)

initial_listing_statusCat=new_X.initial_listing_status.astype("category").cat.codes
initial_listing_statusCat=pd.Series(initial_listing_statusCat)

disbursement_methodCat=new_X.disbursement_method.astype("category").cat.codes
disbursement_methodCat=pd.Series(disbursement_methodCat)

import statsmodels.api as sm

x1 = np.column_stack((new_X['annual_income'],verifiedIncomecat, new_X['debt_to_income'], new_X['debt_to_income_joint'],
                      new_X['earliest_credit_line'],new_X['inquiries_last_12m'],
                      new_X['total_credit_limit'], new_X['months_since_90d_late'], new_X['accounts_opened_24m'], new_X['total_debit_limit'],
                      new_X['num_mort_accounts'], new_X['account_never_delinq_percent'],
                      new_X['loan_amount'], gradecat, subgradecat, loanStatuscat, initial_listing_statusCat, disbursement_methodCat, new_X['balance']))
x1 = sm.add_constant(x1, prepend=True)

X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size = 0.30, shuffle=True)

y_train = np.reshape(y_train, (-1,1))
y_test = np.reshape(y_test, (-1,1))

from sklearn.preprocessing import MinMaxScaler

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_x.fit(X_train)
xtrain_scale=scaler_x.transform(X_train)

scaler_x.fit(X_test)
xtest_scale=scaler_x.transform(X_test)

scaler_y.fit(y_train)
ytrain_scale = scaler_y.transform(y_train)

scaler_y.fit(y_test)
ytest_scale = scaler_y.transform(y_test)

# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam

model = Sequential()

model.add(Dense(1164, activation='relu'))
#model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.2))

model.add(Dense(200, activation='relu'))
#model.add(Dropout(0.2))

model.add(Dense(50, activation='relu'))
#model.add(Dropout(0.2))

model.add(Dense(10, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

custom_optimizer=Adam(learning_rate=0.01)

model.compile(loss='mean_squared_logarithmic_error', optimizer=custom_optimizer, metrics=['mse'])

history = model.fit(xtrain_scale, ytrain_scale, epochs=15, batch_size=100, verbose=1, validation_split=0.2)

predictions = model.predict(xtest_scale)

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

predictions = scaler_y.inverse_transform(predictions)

from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_test, predictions))
rmse

