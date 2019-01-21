import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from  matplotlib import colors

# ideas to remember: "help" the algorithm by filtering differentially expressed genes, create two features
# which will be the avg of the expression data, one for treated and one for non-treated

samples=['GSM1338298', 'GSM1338302', 'GSM1338306', 'GSM1338297', 'GSM1338301', 'GSM1338305', 'GSM1338296', 'GSM1338300',
         'GSM1338304', 'GSM1338295', 'GSM1338299', 'GSM1338303']

# df = pd.read_csv("data.csv")
# df = df.dropna(subset=samples)
# df = df.reset_index()
# print(df)

df = pd.read_csv("no_nan.csv")
gene_exp = df[samples]

#print(gene_exp)

#df.to_csv("no_nan.csv")

model = KMeans(n_clusters=5)
clusters = model.fit_predict(gene_exp)
#silhouette_score = silhouette_score(gene_exp, clusters, metric='euclidean')



# plot clusters in 2D
pca = PCA(n_components=2).fit(gene_exp)
pca_2d = pca.transform(gene_exp)
color_map = colors.ListedColormap(['b', '#681E0E', 'g', '#7F7F7F', '#FF8700']) #, '#FF84E5', '#6600D0',
                               #    'r', 'c', '#D7D7D7', '#F4FF00'])
sc = plt.scatter(pca_2d[:,0],pca_2d[:,1], c=clusters, cmap=color_map)
cb = plt.colorbar(sc, ticks=range(5))
cb.set_label("Cluster number")
cb.set_ticks(range(5))
plt.show()
