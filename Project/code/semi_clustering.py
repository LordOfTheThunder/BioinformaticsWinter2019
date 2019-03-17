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


plt.rcParams['font.size'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['lines.linewidth'] = 2

df = pd.read_csv('data_filtered.csv')
gene_exp = df[samples]

res_kmeans = []
res_hierarchical = []
cluster_range = range(2, 15)
# for i in cluster_range:
#     model = KMeans(n_clusters=i)
#     clusters = model.fit_predict(gene_exp)
#     hier_cluster = AgglomerativeClustering(n_clusters=i)
#     hier_clusters = hier_cluster.fit_predict(gene_exp)
#     res_kmeans.append(metrics.silhouette_score(gene_exp, clusters))
#     res_hierarchical.append(metrics.silhouette_score(gene_exp, hier_clusters))
#
#     # Other metrics which we decided not to use for now
#     # res.append(metrics.normalized_mutual_info_score(df['apoptosis_related'], clusters))
#     # res.append(metrics.homogeneity_score(df['apoptosis_related'], clusters))
#     # res.append(metrics.fowlkes_mallows_score(df['apoptosis_related'], clusters))
#     # res_kmeans.append(metrics.completeness_score(df['apoptosis_related'], clusters))
#     # res_hierarchical.append(metrics.completeness_score(df['apoptosis_related'], hier_clusters))
#     # res_kmeans.append(metrics.davies_bouldin_score(gene_exp, clusters))
#     # res_hierarchical.append(metrics.davies_bouldin_score(gene_exp, hier_clusters))
#     # res_kmeans.append(metrics.calinski_harabaz_score(gene_exp, clusters))
#     # res_hierarchical.append(metrics.calinski_harabaz_score(gene_exp, hier_clusters))
#
# plt.title('Silhouette score as a function of number of clusters')
# plt.plot(cluster_range, res_kmeans, 'r', label='KMeans')
# plt.plot(cluster_range, res_hierarchical, 'g', label='Hierarchical')
# plt.grid(True)
# plt.legend()
# plt.show()

df = pd.read_csv('train_mean.csv')
gene_exp = df[samples]
model = KMeans(n_clusters=3)
model.fit_predict(gene_exp)
test_df = pd.read_csv('test_mean.csv')
test_gene_exp = test_df[samples]
pred = model.predict(test_gene_exp)
pred_dict = Counter(pred)
sorted_pred_dict = sorted(pred_dict.items(), key=operator.itemgetter(1), reverse=True)
print(sorted_pred_dict)

filtered_gene_exp = df[df['label'] == True]
predict_true = list(model.predict(filtered_gene_exp[samples]))
count_list = [predict_true.count(i) for i in range(model.n_clusters)]
pred_dict_true = Counter(predict_true)
sorted_predict_true = sorted(pred_dict_true.items(), key=operator.itemgetter(1), reverse=True)
print(sorted_predict_true)

for size,pred_true in zip(sorted_pred_dict, sorted_predict_true):
    print('Cluster: ', pred_true[0], 'true labels: ', (pred_true[1] / size[1]))

accuracy = max(count_list) / len(predict_true)
print("KMeans accuracy: ", accuracy)
