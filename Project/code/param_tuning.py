from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier

samples=['GSM1338298', 'GSM1338302', 'GSM1338306', 'GSM1338297', 'GSM1338301', 'GSM1338305', 'GSM1338296', 'GSM1338300',
         'GSM1338304', 'GSM1338295', 'GSM1338299', 'GSM1338303']

train = pd.read_csv("train_mean.csv")
X_train = train[samples]
y_train = train['label']

test = pd.read_csv("test_mean.csv")
X_test = test[samples]
y_test = test['label']

#  TODO: create clf
clf =

acc = 0
rec = 0
prec = 0

# kfold
skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
for train_index, test_index in skf.split(X_train, y_train):
    X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

    clf.fit(X_train_fold, y_train_fold)
    y_pred = clf.predict(X_test_fold)


    acc += accuracy_score(y_test_fold, y_pred)
    rec += recall_score(y_test_fold, y_pred)
    prec += precision_score(y_test_fold, y_pred)

print("accuracy score: ", acc/5)
print("recall score: ", rec/5)
print("precision score: ", prec/5)
