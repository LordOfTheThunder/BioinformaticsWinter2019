import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from sklearn import svm

from svm import svm_classifier

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

    # Create a simple classifier
    clf = DecisionTreeClassifier(class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Classify with SVM
    clf = svm.LinearSVC(class_weight='balanced')
    svm = svm_classifier(X_train, y_train, clf)
    y_pred_svm = clf.predict(X_test)

    # Classify with KNN
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    y_pred_knn = clf.predict(X_test)

    print("Decision Tree")
    print("accuracy score: ", accuracy_score(y_test, y_pred))
    print("recall score: ", recall_score(y_test, y_pred))
    print("precision score: ", precision_score(y_test, y_pred))

    print("SVM")
    print("accuracy score: ", accuracy_score(y_test, y_pred_svm))
    print("recall score: ", recall_score(y_test, y_pred_svm))
    print("precision score: ", precision_score(y_test, y_pred_svm))

    print("KNN with N=3")
    print("accuracy score: ", accuracy_score(y_test, y_pred_knn))
    print("recall score: ", recall_score(y_test, y_pred_knn))
    print("precision score: ", precision_score(y_test, y_pred_knn))

   # # res = pd.DataFrame({'true label': y_test.data, 'pred': y_pred})
   # # res.to_csv('res.csv')