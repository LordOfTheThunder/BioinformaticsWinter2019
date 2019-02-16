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

    df = pd.read_csv("data_labeled.csv")
    data = df[df.columns[1:12]]
    labels = df['apoptosis_related']

    # Limit to the two first classes, and split into training and test
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=.3, stratify=labels)

    # Create a simple classifier
    clf = DecisionTreeClassifier(class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Classify with SVM
    clf = svm.LinearSVC(class_weight='balanced')
    svm = svm_classifier(data, labels, clf)
    y_pred_svm = clf.predict(X_test)

    # Classify with KNN
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    y_pred_knn = clf.predict(X_test)

    print("ID3")
    print("score: ", accuracy_score(y_test, y_pred))
    print("recall score: ", recall_score(y_test, y_pred))
    print("accuracy score: ", precision_score(y_test, y_pred))

    print("SVM")
    print("score: ", accuracy_score(y_test, y_pred_svm))
    print("recall score: ", recall_score(y_test, y_pred_svm))
    print("accuracy score: ", precision_score(y_test, y_pred_svm))

    print("KNN with N=3")
    print("score: ", accuracy_score(y_test, y_pred_knn))
    print("recall score: ", recall_score(y_test, y_pred_knn))
    print("accuracy score: ", precision_score(y_test, y_pred_knn))

    res = pd.DataFrame({'true label': y_test.data, 'pred': y_pred})
    res.to_csv('res.csv')