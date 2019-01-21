from sklearn import svm
import pandas as pd
from classifier import abstract_classifier
from sklearn.metrics import accuracy_score

def fix_col_val(string):
    return (string.split('///')[0]).split(':')[1]

def fix_svm_data():
    df = pd.read_csv("svm_data.csv")
    df = df[df['GO:Process ID'].notnull()]
    list = [fix_col_val(row) for row in df['GO:Process ID']]
    df['GO:Process ID'] = list
    df.to_csv("svm_data_fixed.csv", index=False)

class svm_classifier(abstract_classifier):
    def __init__(self, data, labels, clf):
        self.data = data
        self.labels = labels
        self.clf = clf
        clf.fit(data, labels)

    def classify(self, features):
        return self.clf.predict(features)

if __name__ == "__main__":

    fix_svm_data()
    # get data and labels from csv
    df = pd.read_csv("svm_data_fixed.csv")
    data_test = df[df.columns[4:16]]
    rows = len(data_test.index)
    labels = df['GO:Process ID'].head(int(rows * 0.9))
    data = data_test.head(int(rows * 0.9))

    clf = svm.LinearSVC()
    svm = svm_classifier(data, labels, clf)

    test = data_test.tail(int(rows * 0.1))

    y_pred = [svm.classify(features) for features, label in zip(test[0], test[1])]
    y_true = df['GO:Process ID'].tail(int(rows * 0.1))
    accuracy = accuracy_score(y_true, y_pred, normalize=False)
