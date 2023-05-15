import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# import seaborn as sns
import seaborn as sns
df = pd.read_csv("data/Housing_Dataset_Sample.csv")
df2 =pd.read_csv("https://raw.githubusercontent.com/ryanchung403/dataset/main/Housing_Dataset_Sample.csv")
#oberserve
df.head(10)
df.describe().T
sns.displot(df["Price"])
sns.jointplot(x=df['Avg. Area Income'],y=df['Price'])
sns.pairplot(df)
#prepare train
X =df.iloc[:,:5]
y = df['Price']
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test =train_test_split(X,y,test_size=0.3,random_state=54)

#choose model & train
from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train, y_train)
# a=LinearRegression()
# a.fit(X_train,y_train)
# prediction =a.predict(X_test)
#use model
predictions = reg.predict(X_test) 
predictions
y_test

#evaluate model
from sklearn.metrics import r2_score
r2_score(y_test,predictions)
plt.scatter(y_test, predictions, color='purple',alpha=0.2)
