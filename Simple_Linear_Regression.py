#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
df=pd.read_csv("Salary_Data.csv")
X=df.iloc[ : , :-1].values
y=df.iloc[:,1].values

#Splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

#Fitting Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the test set
y_pred=regressor.predict(X_test)

#Visualising the training set result
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.title("Salary Vs Experience(Trainig set)")
plt.show()

#Visualising the training set result
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.title("Salary Vs Experience(Test set)")
plt.show()

