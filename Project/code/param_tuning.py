from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import svm

samples=['GSM1338298', 'GSM1338302', 'GSM1338306', 'GSM1338297', 'GSM1338301', 'GSM1338305', 'GSM1338296', 'GSM1338300',
         'GSM1338304', 'GSM1338295', 'GSM1338299', 'GSM1338303']

train = pd.read_csv("train_oversampled_1.csv")
X_train = train[samples]
y_train = train['label']

test = pd.read_csv("test_norm.csv")
X_test = test[samples]
y_test = test['label']

# kfold
skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
X_train_fold, X_test_fold,  y_train_fold, y_test_fold = [], [], [], []

for train_index, test_index in skf.split(X_train, y_train):
    X_train_fold.append(X_train.iloc[train_index])
    X_test_fold.append(X_train.iloc[test_index])
    y_train_fold.append(y_train[train_index])
    y_test_fold.append(y_train[test_index])

acc_results, recall_results, prec_results = [], [], []
# x_values = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] # max_depth
x_values = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120)
for val in x_values:
    acc = 0
    rec = 0
    prec = 0
    for i in range(0, 5):
        X_train = X_train_fold[i]
        y_train = y_train_fold[i]
        X_test = X_test_fold[i]
        y_test = y_test_fold[i]

       # clf = DecisionTreeClassifier(max_depth=val)
        clf = svm.LinearSVC(max_iter=val)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc += accuracy_score(y_test, y_pred)
        rec += recall_score(y_test, y_pred)
        prec += precision_score(y_test, y_pred)
    print("VAL is: ", val)
    print("accuracy score: ", acc/5)
    print("recall score: ", rec/5)
    print("precision score: ", prec/5)

    acc_results.append(acc/5)
    recall_results.append(rec/5)
    prec_results.append(prec/5)

plt.title('Score of decision tree as a function of max_depth')
plt.grid(True)
plt.xlabel('max_depth')
plt.ylabel('score')
plt.plot(x_values, acc_results, 'r', label="accuracy")
plt.plot(x_values, recall_results, 'b', label="recall")
plt.plot(x_values, prec_results, 'g', label="precision")
plt.legend()
plt.show()
