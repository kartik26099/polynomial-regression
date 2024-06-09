import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
df=pd.read_csv(r"D:\coding journey\aiml\python\udemy\Machine Learning A-Z (Codes and Datasets)\Part 2 - Regression\Section 6 - Polynomial Regression\Python\Position_Salaries.csv")
x=df.iloc[:, 1:-1]
y=df.iloc[:, -1]

pf=PolynomialFeatures(degree=5)
x_poly=pf.fit_transform(x)
print(x_poly)
lr=LinearRegression()
lr.fit(x_poly,y)


#visualization of polynomial feature
plt.scatter(x,y)
plt.plot(x,lr.predict(x_poly))
#preedicting

print(lr.predict(pf.fit_transform([[2]])))