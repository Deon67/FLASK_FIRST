import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

data=pd.read_csv('regression.csv')
print(data.head())

print(data.columns)

x=data[['SAT']]
y=data['GPA']

lin=LinearRegression()
lin.fit(x,y)
print(lin.score(x,y))

pickle.dump(lin,open('linear_model.pkl','wb'))

model=pickle.load(open('linear_model.pkl','rb'))
print(model.predict([[1230]]))
