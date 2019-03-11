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
from sklearn import metrics

samples=['GSM1338298', 'GSM1338302', 'GSM1338306', 'GSM1338297', 'GSM1338301', 'GSM1338305', 'GSM1338296', 'GSM1338300',
         'GSM1338304', 'GSM1338295', 'GSM1338299', 'GSM1338303']

df = pd.read_csv('data_filtered.csv')
gene_exp = df[samples]

res = []
for i in range(2, 8):
    model = KMeans(n_clusters=i)
    clusters = model.fit_predict(gene_exp)
    # res.append(metrics.silhouette_score(gene_exp, clusters))
    # res.append(metrics.normalized_mutual_info_score(df['apoptosis_related'], clusters))
    # res.append(metrics.homogeneity_score(df['apoptosis_related'], clusters))
    # res.append(metrics.fowlkes_mallows_score(df['apoptosis_related'], clusters))
    # res.append(metrics.completeness_score(df['apoptosis_related'], clusters))
    # res.append(metrics.davies_bouldin_score(gene_exp, clusters))
    res.append(metrics.calinski_harabaz_score(gene_exp, clusters))

plt.plot(range(2, 8), res)
plt.show()

# test_df = pd.read_csv('test_mean.csv')
# test_gene_exp = test_df[samples]
# pred = model.predict(test_gene_exp)
# pred_dict = Counter(pred)
# sorted_pred_dict = sorted(pred_dict.items(), key=operator.itemgetter(1), reverse=True)
# print(sorted_pred_dict)