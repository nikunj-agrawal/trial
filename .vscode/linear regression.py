import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()

data = pd.read_csv('C:/Users/Nikunj Agrawal/Downloads/[FreeCourseSite.com] Udemy - The Data Science Course 2019 Complete Data Science Bootcamp/32. Advanced Statistical Methods - Linear regression with StatsModels/real_estate_price_size.csv')
y = data['price']
x1 = data['size']
x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()

plt.scatter(x1,y)
yhat = x1*223.1787+101900
fig = plt.plot(x1,yhat, lw=4, c='orange', label ='regression line')
plt.xlabel('Size', fontsize = 20)
plt.ylabel('Price', fontsize = 20)
plt.show()