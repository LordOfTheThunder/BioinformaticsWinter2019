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

samples=['GSM1338298', 'GSM1338302', 'GSM1338306', 'GSM1338297', 'GSM1338301', 'GSM1338305', 'GSM1338296', 'GSM1338300',
         'GSM1338304', 'GSM1338295', 'GSM1338299', 'GSM1338303']

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