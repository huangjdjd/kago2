import joblib
model_pretrained =joblib.load('Titanic-LR_20230327.pkl')
import pandas as pd
df_test =pd.read_csv("https://raw.githubusercontent.com/ryanchung403/dataset/main/train_data_titanic.csv")
df_test.drop(['Name','Ticket'],axis=1,inplace=True)
df_test.drop('Cabin',axis=1,inplace=True)
df_test.info()
df_test['Age']=df_test.groupby('Sex')['Age'].apply(lambda x:x.fillna(x.median()))
df_test.isnull().sum()
df_test['Fare'].value_counts()
df_test['Fare'].fillna(df_test['Fare'].value_counts().idxmax(),inplace=True)
df_test=pd.get_dummies(data=df_test,columns=['Sex','Embarked'])
df_test.drop('Sex_female',axis=1,inplace=True)
df_test.drop('Pclass',axis=1,inplace=True)
predictions2=model_pretrained.predict(df_test)
predictions2
forSubmissionDF =pd.DataFrame(columns=['PassengerId','Survived'])
forSubmissionDF
forSubmissionDF['PassengerId']=range(892,1310)
forSubmissionDF['Survived']=predictions2
forSubmissionDF.to_csv('for_submission_20230327.csv',index=False)