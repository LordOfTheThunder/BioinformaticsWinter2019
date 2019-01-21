from sklearn import svm
import pandas as pd
from classifier import abstract_classifier

def fix_col_val(string):
    if pd.isnull(string):
        return 0
    return (string.split('///')[0]).split(':')[1]

def fix_svm_data():
    df = pd.read_csv("svm_data.csv")
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

    def accuracy():
        return (true_pos + true_neg) / N

    def error():
        return (false_pos + false_neg) / N

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

    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    N = len(test[0])
    for features, label in zip(test[0], test[1]):
            res = svm.classify(features)
            if res == 0 and label == 0:
                true_neg += 1
            if res == 0 and label == 1:
                false_neg += 1
            if res == 1 and label == 0:
                false_pos += 1
            if res == 1 and label == 1:
                true_pos += 1

    print(accuracy(), error())