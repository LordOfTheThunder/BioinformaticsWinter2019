from sklearn import svm
import pandas as pd
from classifier import abstract_classifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

def fix_col_val(string):
    s = string.partition('///')[0]
    s = s.partition(':')[2]
    if s:
        return s
    else:
        return string

# 0005515: protein binding
# 0046872: metal ion binding
# 0044822: RNA binding (same as 0003723)
# 0005524: ATP binding
# 0003677: DNA binding
# 0005509: calcium ion binding
# 0003674: molecular_function
# 0004930: G protein-coupled receptor activity
# 0005525: GTP binding
# 0003723: RNA binding

labels = ['0005525', '0044822', '0046872', '0003723', '0003674', '0004930', '0005509', '0003677',
          '0005524','0005515', '0005515']

def fix_svm_data():
    df = pd.read_csv("no_nan.csv")
    df = df[df['GO:Function ID'].notnull()]

    list = []
    for row in df['GO:Function ID']:
        new_val = row
        if '0005525' in row:
            new_val = '0005525'
        elif '0044822' in row:
            new_val = '0044822'
        elif '0046872' in row:
            new_val = '0046872'
        elif '0003723' in row:
            new_val = '0003723'
        elif '0003674' in row:
            new_val = '0003674'
        elif '0004930' in row:
            new_val = '0004930'
        elif '0005509' in row:
            new_val = '0005509'
        elif '0003677' in row:
            new_val = '0003677'
        elif '0005524' in row:
            new_val = '0005524'
        elif '0005515' in row:
            new_val = '0005515'
        list.append(new_val)

    print(list)

    df['GO:Function ID'] = list
   # list_ = [fix_col_val(row) for row in df['GO:Function ID']]
  #  df['GO:Function ID'] = list_
    #df = df[~df['GO:Function ID'].str.contains('///')]
    df = df[df['GO:Function ID'].isin(labels)]

    from collections import Counter
    print(Counter(df['GO:Function ID']))
    return df

    #df.to_csv("svm_data_fixed.csv", index=False)

class svm_classifier(abstract_classifier):
    def __init__(self, data, labels, clf):
        self.data = data
        self.labels = labels
        self.clf = clf
        clf.fit(data, labels)

    def classify(self, features):
        return self.clf.predict(features)

if __name__ == "__main__":

    df = fix_svm_data()
    # get data and labels from csv
   # df = pd.read_csv("svm_data_fixed.csv")
    data_test = df[df.columns[4:16]]
    rows = len(data_test.index)
    labels = df['GO:Function ID'].head(int(rows * 0.9))
    data = data_test.head(int(rows * 0.9))



    clf = DecisionTreeClassifier()
    clf.fit(data, labels)

    print(clf.score(data_test.tail(int(rows * 0.1)), df['GO:Function ID'].tail(int(rows * 0.1))))

    # clf = svm.LinearSVC()
    # svm = svm_classifier(data, labels, clf)
    # test = data_test.tail(int(rows * 0.1))
    # y_pred = [svm.classify(features) for features, label in zip(test[0], test[1])]
    #y_true = df['GO:Function ID'].tail(int(rows * 0.1))
    #accuracy = accuracy_score(y_true, y_pred, normalize=False)
   # print(accuracy)
