import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import numpy as np
from sklearn import svm

samples=['GSM1338298', 'GSM1338302', 'GSM1338306', 'GSM1338297', 'GSM1338301', 'GSM1338305', 'GSM1338296', 'GSM1338300',
         'GSM1338304', 'GSM1338295', 'GSM1338299', 'GSM1338303']

if __name__ == "__main__":
    # warnings.filterwarnings("ignore")
    #
    # df = pd.read_csv("data_labeled.csv")
    # data = df[df.columns[0:13]]
    #
    # print(data)
    #
    #
    # labels = df['apoptosis_related']
    #
    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=.3, stratify=labels)
    #
    # X_train = X_train.assign(label = y_train)
    # X_train.to_csv('train_probes.csv')
    #
    # X_test = X_test.assign(label = y_test)
    # X_test.to_csv('test_probes.csv')

    train = pd.read_csv("train_mean.csv")
    X_train = train[samples]
    y_train = train['label']

    test = pd.read_csv("test_mean.csv")
    X_test = test[samples]
    y_test = test['label']

    # Standardize
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # log-scale
    # log_transform = FunctionTransformer(np.log1p)
    # X_train = log_transform.transform(X_train)
    # X_test = log_transform.transform(X_test)

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

   # # res = pd.DataFrame({'true label': y_test.data, 'pred': y_pred})
   # # res.to_csv('res.csv')