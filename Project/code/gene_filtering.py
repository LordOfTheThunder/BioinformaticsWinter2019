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
    # train = pd.read_csv("train_mean.csv")
    # test = pd.read_csv("test_mean.csv")

    data = pd.read_csv("data_labeled.csv")

    # Perform normalization
    # data[samples] = StandardScaler().fit_transform(data[samples])
    # train[samples] = scaler.transform(train[samples])
    # test[samples] = scaler.transform(test[samples])

    # OR: Apply min-max scaler (comment out the unnecessary one)
    # scaler = MinMaxScaler(feature_range=(0, 1)).fit(train[samples])
    # train[samples] = scaler.transform(train[samples])
    # test[samples] = scaler.transform(test[samples])

    # compute variance of each row
    # var = train.var(axis=1, numeric_only=True)
    # train['var'] = var
    #
    # var = test.var(axis=1, numeric_only=True)
    # test['var'] = var
   # print(train)
    var = data.var(axis=1, numeric_only=True)
    data['var'] = var
    print(data)

    # print("before filtering, train size is: ", len(train), " test size is: ", len(test))

    # train = train[train['var'] > 0.005]
    # test = test[test['var'] > 0.005]

    data = data[data['var'] > .2e7]

    # print("after filtering, train size is: ", len(train), " test size is: ", len(test))
    # print("TRUE in train size: ", len(train[train['label'] == True]),
    #       "TRUE in test size: ", len(test[test['label'] == True]))

    labels = data['apoptosis_related']
    X_train, X_test, y_train, y_test = train_test_split(data[samples], labels, test_size=.3, stratify=labels)

    print("after filtering, train size is: ", len(X_train), " test size is: ", len(X_test))

    # Create decision tree
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Decision Tree")
    print("accuracy score: ", accuracy_score(y_test, y_pred))
    print("recall score: ", recall_score(y_test, y_pred))
    print("precision score: ", precision_score(y_test, y_pred))

    # X_train = X_train.assign(label = y_train)
    # X_train.to_csv('train_filtered.csv')
    #
    # X_test = X_test.assign(label = y_test)
    # X_test.to_csv('test_filtered.csv')


