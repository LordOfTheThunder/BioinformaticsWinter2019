import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from  matplotlib import colors
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from collections import Counter, defaultdict
import operator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

samples=['GSM1338298', 'GSM1338302', 'GSM1338306', 'GSM1338297', 'GSM1338301', 'GSM1338305', 'GSM1338296', 'GSM1338300',
         'GSM1338304', 'GSM1338295', 'GSM1338299', 'GSM1338303']

def low_variance_filter(train, test):
    None

"""
df = pd.read_csv('data_labeled.csv')
gene_exp = df[samples]
model = KMeans(n_clusters=5)
clusters = model.fit(gene_exp)
# Filter 12,000 genes from 0 cluster
cnt = 0
for idx, gene in enumerate(df['IDENTIFIER']):
    pred = model.predict([gene_exp.iloc[idx]])
    if cnt == 12000:
        break
    if pred == 0:
        cnt += 1
        df = df.drop(idx, axis=0)
df.to_csv('data_labeled_undersampled.csv')
"""


def calc_clusters():
    # Calculating new clustering
    df = pd.read_csv('data_labeled_undersampled.csv')
    gene_exp = df[samples]
    model = KMeans(n_clusters=5)
    clusters = model.fit_predict(gene_exp)
    res_dict = Counter(clusters)
    sorted_res_dict = sorted(res_dict.items(), key=operator.itemgetter(1), reverse=True)
    print('Elements per cluster')
    print(sorted_res_dict)

    filtered_gene_exp = df[df['apoptosis_related'] == True]
    predict_true = list(model.predict(filtered_gene_exp[samples]))
    count_list = [predict_true.count(i) for i in range(model.n_clusters)]
    accuracy = max(count_list) / len(predict_true)
    print("KMeans accuracy: ", accuracy)


if __name__ == '__main__':
    train = pd.read_csv("train_mean.csv")
    test = pd.read_csv("test_mean.csv")

    # Perform normalization
    scaler = StandardScaler().fit(train[samples])
    train[samples] = scaler.transform(train[samples])
    test[samples] = scaler.transform(test[samples])

    X_test = test[samples]
    y_test = test['label']

    # compute variance of each row
    var = train[samples].var(axis=1, numeric_only=True)
    train['var'] = var
    print(train)

    print(test)

    print("before filtering, train size is: ", len(train), " test size is: ", len(test))
    acc_results, recall_results, prec_results = [], [], []
    x_values = [0.001, 0.005, 0.01, 0.05, 0.07, 0.08]
    for val in x_values:
        train_tmp = train[train['var'] > val]

        print("after filtering, train size is: ", len(train), " test size is: ", len(test))
        print("TRUE in train size: ", len(train[train['label'] == True]),
              "TRUE in test size: ", len(test[test['label'] == True]))

        X_train = train_tmp[samples]
        y_train = train_tmp['label']
        # Create decision tree
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc_score = accuracy_score(y_test, y_pred)
        rec_score = recall_score(y_test, y_pred)
        prec_score = precision_score(y_test, y_pred)

        print("Decision Tree")
        print("accuracy score: ", acc_score)
        print("recall score: ", rec_score)
        print("precision score: ", prec_score)

        acc_results.append(acc_score)
        recall_results.append(rec_score)
        prec_results.append(prec_score)

    plt.title('Score as a function of filtering variance')
    plt.grid(True)
    plt.xlabel('variance filter')
    plt.ylabel('score')
    plt.plot(x_values, acc_results, 'r', label="accuracy")
    plt.plot(x_values, recall_results, 'b', label="recall")
    plt.plot(x_values, prec_results, 'g', label="precision")
    plt.legend()
    plt.show()


