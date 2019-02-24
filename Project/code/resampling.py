import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, precision_score
from math import ceil

samples=['GSM1338298', 'GSM1338302', 'GSM1338306', 'GSM1338297', 'GSM1338301', 'GSM1338305', 'GSM1338296', 'GSM1338300',
         'GSM1338304', 'GSM1338295', 'GSM1338299', 'GSM1338303']

def undersample(df, coef):
    resample_indices = df[df['label'] == False].index # examples we need to make less of
    rest_indices = df[df['label'] == True].index
    n_u = ceil(len(resample_indices) * coef) # amount that stays is coef * number of 'False' indices
    resampled_indices = np.random.choice(resample_indices, n_u, replace=False)
    final_indices = np.concatenate([rest_indices, resampled_indices])
    return df.iloc[final_indices] #list(itertools.compress(undersampled_indices, y_train))

def oversample(df, coef):
    resample_indices = df[df['label'] == True].index # examples we need to make less of
    rest_indices = df[df['label'] == False].index
    n_u = ceil(len(rest_indices) * coef) # coef is how much 'True' we want compared to 'False'
    resampled_indices = np.random.choice(resample_indices, n_u, replace=True)
    final_indices = np.concatenate([rest_indices, resampled_indices])
    return df.iloc[final_indices] #list(itertools.compress(undersampled_indices, y_train))


train = pd.read_csv("train_mean.csv")
#train = undersample(train, 0.13)
train = oversample(train, 0.9)

print("resampled train length: ", len(train))

print("after resampling, train size is: ", len(train))
print("TRUE in train size: ", len(train[train['label'] == True]))

X_train = train[samples]
y_train = train['label']

test = pd.read_csv("test_mean.csv")
X_test = test[samples]
y_test = test['label']

# Create decision tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Create random forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

ab = AdaBoostClassifier()
ab.fit(X_train, y_train)
y_pred_ab = ab.predict(X_test)

# Classify with SVM
clf = svm.LinearSVC()
clf.fit(X_train, y_train)
y_pred_svm = clf.predict(X_test)

clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred_svc = clf.predict(X_test)

# Classify with KNN
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
y_pred_knn = clf.predict(X_test)

print("Decision Tree")
print("accuracy score: ", accuracy_score(y_test, y_pred))
print("recall score: ", recall_score(y_test, y_pred))
print("precision score: ", precision_score(y_test, y_pred))

print("Random Forest")
print("accuracy score: ", accuracy_score(y_test, y_pred_rf))
print("recall score: ", recall_score(y_test, y_pred_rf))
print("precision score: ", precision_score(y_test, y_pred_rf))

print("AdaBoost")
print("accuracy score: ", accuracy_score(y_test, y_pred_ab))
print("recall score: ", recall_score(y_test, y_pred_ab))
print("precision score: ", precision_score(y_test, y_pred_ab))

print("SVM")
print("accuracy score: ", accuracy_score(y_test, y_pred_svm))
print("recall score: ", recall_score(y_test, y_pred_svm))
print("precision score: ", precision_score(y_test, y_pred_svm))

print("SVC")
print("accuracy score: ", accuracy_score(y_test, y_pred_svc))
print("recall score: ", recall_score(y_test, y_pred_svc))
print("precision score: ", precision_score(y_test, y_pred_svc))

print("KNN with N=3")
print("accuracy score: ", accuracy_score(y_test, y_pred_knn))
print("recall score: ", recall_score(y_test, y_pred_knn))
print("precision score: ", precision_score(y_test, y_pred_knn))