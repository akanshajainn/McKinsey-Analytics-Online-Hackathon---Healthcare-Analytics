'''This is just a working python code. It takes 'train.csv' and 'test.csv' as input from the local directory and outputs 'samplesubmission.csv'.
Please refer Akansha_Submission_Explanation.ipynb notebook for complete explanation of the technique used along with visualisation. '''

import pandas as pd
import numpy as np
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from catboost import CatBoostClassifier
from catboost import CatBoost
import catboost as ctb

df=pd.read_csv('train.csv')

del df['id']
del df['ever_married']

newbmi=np.where(df['bmi'].isnull(),
                df['bmi'].mean(),
                df['bmi'])
df['bmi']=newbmi


newss=np.where(df['smoking_status'].isnull(),
                "unknown",
                df['smoking_status'])
df['smoking_status']=newss

index = np.where(df["avg_glucose_level"] == max(df["avg_glucose_level"]) )
df=df.drop(index[0])

df=df[df['bmi']<60]

df['gender']=df['gender'].astype('category')
df['Residence_type'] = df['Residence_type'].astype('category')

df['gender']=df['gender'].cat.codes
df['Residence_type']=df['Residence_type'].cat.codes

df=pd.get_dummies(df, columns=['work_type','smoking_status'], prefix=[ "work",'smoking'])

X = df.loc[:, df.columns != 'stroke']
Y = df['stroke']

seed = 7
test_size = 0.33
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=test_size, random_state=seed)

catmodel=CatBoostClassifier()
catmodel.fit(X_train, y_train)

y_pred_cat = catmodel.predict(X_val)

accuracycat = accuracy_score(y_val, y_pred_cat)

print("Accuracy: %.2f%%" % (accuracycat * 100.0))

print("AUC ROC: %.2f%%" % roc_auc_score(y_val, y_pred_cat))

test=pd.read_csv('test.csv')

del test['ever_married']

newss=np.where(test['smoking_status'].isnull(),
                "unknown",
                test['smoking_status'])
test['smoking_status']=newss

newbmi=np.where(test['bmi'].isnull(),test['bmi'].mean(),test['bmi'])
test['bmi']=newbmi

test['gender']=test['gender'].astype('category')
test['Residence_type'] = test['Residence_type'].astype('category')
test['gender']=test['gender'].cat.codes
test['Residence_type']=test['Residence_type'].cat.codes

test=pd.get_dummies(test, columns=['work_type','smoking_status'], prefix=[ "work",'smoking'])

X_test = test.loc[:, test.columns != 'id']
testid=test['id']

Y_test_label_cat= catmodel.predict(X_test)
Y_test_prob_cat= catmodel.predict_proba(X_test)

Y_test_cat=[None]*len(X_test)
for i in range(len(X_test)):
    Y_test_cat[i]=Y_test_prob_cat[i][1]

sample=pd.DataFrame()
sample['id']=testid
sample['stroke']=Y_test_cat
sample.to_csv('samplesubmission.csv', index=False)