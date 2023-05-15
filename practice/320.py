import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df =pd.read_csv("https://raw.githubusercontent.com/ryanchung403/dataset/main/train_data_titanic.csv")
df.head()
df.describe().T
df.info()
# axis指的是行 如果是列要0 implace取代要true
df.drop(['Name','Ticket'],axis=1,inplace=True)
sns.pairplot(df[['Survived','Fare']],dropna=True)
sns.pairplot(df[['Survived','Age']],dropna=True)
df.groupby('Survived').mean()
sns.displot(df['Survived'])
df['SibSp'].value_counts
df.isnull().sum()
len(df)/2
# True False
df.isnull().sum()>(len(df)/2) 
df.drop('Cabin',axis=1,inplace=True)
df['Age'].isnull().value_counts()
df.groupby('Sex')['Age'].median().plot(kind='bar')
df['Age'] = df.groupby('Sex')['Age'].apply(lambda x: x.fillna(x.median()))
df['Embarked'].value_counts().idxmax()
df['Embarked'].fillna(df['Embarked'].value_counts().idxmax(),inplace=True)
df =pd.get_dummies(data=df,columns=['Sex','Embarked'])
df
df.drop('Sex_female',axis=1,inplace=True)
df.info()
df.corr()
X =df.drop(['Survived','Pclass'],axis=1)
y= df['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=67)
from sklearn.linear_model import LogisticRegression
lr =LogisticRegression(max_iter=200)
lr.fit(X_train,y_train)

predictions =lr.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score
accuracy_score(y_test,predictions)
recall_score(y_test,predictions)
precision_score(y_test,predictions)

confusion_matrix(y_test,predictions)
pd.DataFrame(confusion_matrix(y_test,predictions),columns=['Predictnot Survived', 'PredictSurvived'],index=['Truenot Survived','TrueSurvived'])

import joblib
joblib.dump(lr,'Titanic-LR_20230327',compress=3)
