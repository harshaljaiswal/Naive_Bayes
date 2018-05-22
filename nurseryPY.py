# Naive Bayes

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
name = ['parents','has_nurs','form','children','housing','finance','social','health','class']  
dataset = pd.read_csv('nurseryNVB.csv', names = name, header = None)
data = dataset.iloc[:,:]

# encoding the categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label = LabelEncoder()
# doing for all the columns of data
for i in range(9):
    data.iloc[:,i] = label.fit_transform(data.iloc[:,i])

# one hot encoding for the categorical variables    
onehotencoder = OneHotEncoder()
x= data.iloc[:,:-1].values
y = data.iloc[:,8].values

onehotencoder = OneHotEncoder(categorical_features = [0,1,2,3,4,5,6,7])
x = onehotencoder.fit_transform(x).toarray()

# saving our model from dummy variable trap
x = x[:,[1,2,4,5,6,7,9,10,11,13,14,15,17,18,20,22,23,25,26]]

# Partitioning the training and testset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0, stratify = y)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Making the accuracy Matrix
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
# ac = 0.766358024691358
