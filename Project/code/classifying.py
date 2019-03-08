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
    df = pd.read_csv("data_labeled.csv")
    data = df[df.columns[0:13]]

    labels = df['apoptosis_related']

    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=.15, stratify=labels)
    #
    # X_train = X_train.assign(label = y_train)
    # X_train.to_csv('train_mean_2.csv')
    #
    # X_test = X_test.assign(label = y_test)
    # X_test.to_csv('test_mean_2.csv')

    # train = pd.read_csv("train_undersampled_05_.csv")
    # train = pd.read_csv("train_undersampled_05_.csv")
    train = pd.read_csv("train_oversampled_1.csv")
    test = pd.read_csv("test_norm.csv")
    # train = pd.read_csv("train_mean.csv")
    # test = pd.read_csv("test_mean.csv")

    diff_1 = train[samples[0]] - train[samples[3]]
    diff_2 = train[samples[1]] - train[samples[4]]
    diff_3 = train[samples[2]] - train[samples[5]]
    diff_4 = train[samples[6]] - train[samples[9]]
    diff_5 = train[samples[7]] - train[samples[10]]
    diff_6 = train[samples[8]] - train[samples[11]]

    diff_1_test = test[samples[0]] - test[samples[3]]
    diff_2_test = test[samples[1]] - test[samples[4]]
    diff_3_test = test[samples[2]] - test[samples[5]]
    diff_4_test = test[samples[6]] - test[samples[9]]
    diff_5_test = test[samples[7]] - test[samples[10]]
    diff_6_test = test[samples[8]] - test[samples[11]]

    data = {'diff_1': diff_1, 'diff_2': diff_2, 'diff_3': diff_3, 'diff_4': diff_4, 'diff_5': diff_5, 'diff_6': diff_6}
    X_train = pd.DataFrame.from_dict(data)

    # X_train = train[samples]
    y_train = train['label']

    data_ = {'diff_1': diff_1_test, 'diff_2': diff_2_test, 'diff_3': diff_3_test, 'diff_4': diff_4_test,
             'diff_5': diff_5_test, 'diff_6': diff_6_test}
    X_test = pd.DataFrame.from_dict(data_)

    # X_test = test[samples]
    y_test = test['label']

    # Standardize
    # scaler = StandardScaler().fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)

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