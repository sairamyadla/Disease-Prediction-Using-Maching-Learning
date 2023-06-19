###Importing Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz #plot tree
from sklearn.metrics import roc_curve, auc #for model evaluation
from sklearn.metrics import classification_report #for model evaluation
from sklearn.metrics import confusion_matrix #for model evaluation
from sklearn.model_selection import train_test_split #for data splitting
from sklearn.metrics import r2_score,accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.tree import export_graphviz #plot tree
from sklearn.metrics import roc_curve, auc #for model evaluation
from sklearn.metrics import classification_report #for model evaluation
from sklearn.metrics import confusion_matrix #for model evaluation

import pylab

# import tensorflow and keras
import tensorflow as tf
from tensorflow.keras.models import Sequential   # used for initialize ANN model
from tensorflow.keras import layers   # used for different layer structure
from tensorflow.keras.layers import Dense

import pylab
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

## Uploading dataset files..

train_set = pd.read_csv(r"/content/Training.csv",sep =",")
test_set = pd.read_csv(r"/content/Testing.csv",sep =",")
train_set = train_set.iloc[:,:-1]
test_set.head()
------------------------------------------------------------------------------------
nRow,nCol=train_set.shape
print(f'**Summary**:\n There are {nRow} rows and {nCol} columns. prognosis is the target/label variable.')

##**Summary**:
 There are 4920 rows and 133 columns. prognosis is the target/label variable.
  
-----------------------------------------------------------------
train_set['prognosis'].value_counts()
-------------------------------------------------------------
# Get the number of missing data points, NA's ,NAN's values per column
total = train_set.isnull().sum().sort_values(ascending=False)
percent = (train_set.isnull().sum()/train_set.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

total = train_set.isna().sum().sort_values(ascending=False)
percent = (train_set.isna().sum()/train_set.isna().count()).sort_values(ascending=False)
na_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

if((na_data.all()).all()>0 or (na_data.all()).all()>0):
     print('Found Missing Data or NA values')
        
else:
    print('There is no missing data or null values in the collected data. Additionally, the length of each column is same.')
    
-----------------------------------------------------------------------------------
##correlation matrix
corr_matrix=train_set.corr()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper
---------------------------------------------------------------------------------------
# using sklearn variance threshold to find constant features
temp_train=train_set.iloc[:,:-1]

from sklearn.feature_selection import VarianceThreshold


sel = VarianceThreshold(threshold=0.03)
sel.fit(temp_train) ##0.03
-------------------------------------------------------------------------------
### training and splitting the data.
encoder = LabelEncoder()
train_set["prognosis"] = encoder.fit_transform(train_set["prognosis"])
test_set["prognosis"] = encoder.transform(test_set["prognosis"])
X_train, X_valid, y_train, y_valid = train_test_split(train_set.drop('prognosis', 1), train_set['prognosis'], test_size = .4,
                                                      random_state=42,shuffle=True)
X_train.shape ### (2952, 49)
test_set = pd.concat([test_set,pd.concat([X_valid,y_valid],axis=1)],axis=0)
test_set.shape ##(2010, 50)
-------------------------------------------------------------
##Logistic Regression
lr=LogisticRegression(C=0.2,random_state=42, penalty='l2')
lr.fit(X_train,y_train)
print("Logistic Train score with ",format(lr.score(X_train, y_train))) ##0.9847560975609756
print("Logistic Test score with ",format(lr.score(test_set.iloc[:,:-1], test_set['prognosis']))) ##0.9835820895522388
------------------------------------------------------------------------
##Decision Trees

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print("Decision Tree Train score with ",format(dt.score(X_train, y_train))) ##0.9854336043360433
print("Desicion Tree Test score with ",format(dt.score(test_set.iloc[:,:-1], test_set['prognosis']))) ##0.982089552238806
-------------------------------------------------------------------------
##Random Forest

rf = RandomForestClassifier(max_depth=6,oob_score=True,random_state=42,criterion='entropy',max_features='auto',n_estimators=300)
rf.fit(X_train, y_train)
y_pred=rf.predict(X_valid)
print("Random Forest Train score with ",format(rf.score(X_train, y_train))) #0.9773035230352304
print("Random Forest Test score with ",format(rf.score(test_set.iloc[:,:-1], test_set['prognosis']))) #0.9796019900497512
----------------------------------------------------------------------------------
##SVM

svm = SVC()
svm.fit(X_train, y_train)
y_pred=svm.predict(X_valid)
print("SVM Train score with ",format(svm.score(X_train, y_train))) #0.9854336043360433
print("SVM Test score with ",format(svm.score(test_set.iloc[:,:-1], test_set['prognosis']))) #0.9855721393034826
-------------------------------------------------------------------------------
##Naive BAyes

bayes = GaussianNB()
bayes.fit(X_train, y_train)
y_pred=bayes.predict(X_valid) 
print("Naive Bayes Train score with ",format(bayes.score(X_train, y_train))) # 0.967140921409214
print("Naive Bayes Test score with ",format(bayes.score(test_set.iloc[:,:-1], test_set['prognosis'])),'%') # 0.9671641791044776 
--------------------------------------------------------------------------------
##ANN

# transform into dummies for y_train (prognosis variable)
y_train_dum = pd.get_dummies(y_train)
classifier = Sequential()

classifier.add(Dense(64, activation = "relu", input_dim = X_train.shape[1]))
# adding second hidden layer
classifier.add(Dense(48, activation = "relu"))
# adding last layer
classifier.add(Dense(y_train_dum.shape[1], activation = "softmax"))

classifier.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
classifier.summary()


history = classifier.fit(X_train, y_train_dum, epochs = 5, batch_size = 30)

print("ANN Train score with ",format(history.history['accuracy'][-1])) #0.9844173192977905

prediction = classifier.predict(test_set.iloc[:,:-1])

prediction = [np.argmax(i) for i in prediction ]

print("ANN Test score with ",format(accuracy_score(test_set['prognosis'], prediction)),'%') # 0.9855721393034827 
---------------------------------------------------------------------------------
##KNN
error = []
# Calculating error for K values between 1 and 30
for i in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_valid)
    error.append(np.mean(pred_i != y_valid))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 30), error, color='red', linestyle='dashed', marker='*',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
print("Minimum error:-",min(error),"at K =",error.index(min(error))+1) #0.014735772357723578 
print(" KNN test score with",format(accuracy_score(y_valid, y_pred))) #0.9852642276422764
--------------------------------------------------------------------------------
##Bagging & Boosting

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

bg = BaggingClassifier( max_samples=1.0,max_features=1.0, n_estimators=50)
bg.fit(X_train,y_train) 
bg.score(X_valid,y_valid)


bg.score(X_train,y_train)

print("Bagging Train score with ",format(bg.score(X_train, y_train))) #0.9854336043360433


ada = AdaBoostClassifier(n_estimators=10,learning_rate=1)
ada.fit(X_train,y_train)
ada.score(X_valid,y_valid)
model=ada.fit(X_train,y_train)
#predict the response for test dataset
y_pred=model.predict(X_valid)
print("Boosting Train score with ",format(ada.score(X_train, y_train))) #0.15176151761517614
print("Bagging Test score with ",format(bg.score(X_valid, y_valid))) #0.9852642276422764
print("Boosting Test score with ",format(ada.score(X_valid, y_pred))) #1.0
---------------------------------------------------------------------------------
##Comparision of all classification techniques
sns.set(style="darkgrid")

acc = pd.DataFrame({'Model':['Logistic','Decision Tree','Random Tree','SVM','Naive Byes','ANN','KNN','Bagging','Boosting'],
                    'Accuracy':[98.3,98.2,97.9,98.5,96.7,98.3,98.5,98.5,100]})

# Set the figure size
fig, ax =  plt.subplots(figsize=(12, 9))
ax.set_ylim(96.5, 100)
# plot a bar chart
sns.barplot(
    x="Model", 
    y="Accuracy", 
    data=acc,  
    ci=None, 
    color='#69b3a2',
    orient = 'v');
plt.show()
