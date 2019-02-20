import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from  matplotlib import colors
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import operator

if __name__ == "__main__":
    df = pd.read_csv('data_labeled_no_mean.csv')
    apoptosis_related = df['apoptosis_related']
    go_terms = df['GO:Process ID']
    df = pd.read_csv('terms.csv')
    apoptosis_related_terms = df['id']
    go_term_counter = {}
    for terms,is_related in zip(go_terms, apoptosis_related):
        if is_related is False:
            continue
        for term in terms.split('///'):
            if term not in list(apoptosis_related_terms):
                continue
            go_term_counter[term] = 1 if term not in go_term_counter else go_term_counter[term] + 1

    sorted_terms = sorted(go_term_counter.items(), key=operator.itemgetter(1), reverse=True)
    x, y = zip(*sorted_terms)
    plt.title('Go term relevance')
    plt.xlabel('Go terms')
    plt.ylabel('Count')
    plt.plot(x, y)
    plt.show()

    res = pd.DataFrame(sorted_terms, columns=["go_term", "count"])
    res.to_csv('go_term_relevance.csv')