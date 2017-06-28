import numpy as np
import pandas as pd 
    
#Importing Datasets
#2,4,5,6,7,9
dataset = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X = dataset.loc[:,['Pclass','Sex','Age','SibSp','Parch','Fare']].values
y = dataset.iloc[:,1].values
X_test = test.loc[:,['Pclass','Sex','Age','SibSp','Parch','Fare']].values

#Taking Care of Missing Values
from sklearn.preprocessing import Imputer
imputer_age = Imputer(missing_values='NaN', strategy='median',axis=0)
X[:,2:3] = imputer_age.fit_transform(X[:,2:3])
X_test[:,2:3] = imputer_age.transform(X_test[:,2:3])
imputer_fare = Imputer(missing_values='NaN', strategy='median',axis=0)
X[:,5:6] = imputer_fare.fit_transform(X[:,5:6])
X_test[:,5:6] = imputer_fare.transform(X_test[:,5:6])

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_gender = LabelEncoder()
X[:,1] = labelencoder_gender.fit_transform(X[:,1])
X_test[:,1] = labelencoder_gender.transform(X_test[:,1])

#Converting dtype to float
X=X.astype(dtype='float')
X_test = X_test.astype(dtype='float')

#One Hot Encoding Categorical Features to create dummy variables
onehotencoder_class = OneHotEncoder(categorical_features = [0])
X = onehotencoder_class.fit_transform(X).toarray() 
X_test = onehotencoder_class.transform(X_test).toarray()

#Preventing dummy variable trap
X= X[:,1:]
X_test =X_test[:,1:]

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X= sc.fit_transform(X)
X_test = sc.transform(X_test)

#Initialising and fitting model and dataset respectively
from sklearn.svm import SVC
classifier = SVC(random_state=123)
classifier.fit(X,y)
  
#Creating prediction result CSV
y_pred = classifier.predict(X_test)
pass_id = test.iloc[:,0].values
pass_id= pass_id.reshape(418,1)
submission = np.concatenate((pass_id, y_pred.reshape(418,1)),axis=1)
np.savetxt("Predictions.csv", X=submission, delimiter=",")
