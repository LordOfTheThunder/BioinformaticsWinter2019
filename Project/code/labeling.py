import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from  matplotlib import colors

samples=['GSM1338298', 'GSM1338302', 'GSM1338306', 'GSM1338297', 'GSM1338301', 'GSM1338305', 'GSM1338296', 'GSM1338300',
         'GSM1338304', 'GSM1338295', 'GSM1338299', 'GSM1338303']

# unify all rows with same gene into one row, expression levels are the mean
# df = pd.read_csv("no_nan.csv")
# df = df.sort_values('IDENTIFIER')
# df2 = df.drop_duplicates(subset='IDENTIFIER')
# df1 = df.groupby('IDENTIFIER', as_index=False)[samples].mean()
# ~ here I unified the dataframes in excel ~

df = pd.read_csv("no_dup.csv")
df = df.dropna(subset=['GO:Process ID'])
columns = samples
columns.append('GO:Process')
columns.append('GO:Process ID')
df = df[columns]
print(df)

terms = pd.read_csv("terms.csv")

is_apoptosis = []
for row in df['GO:Process ID']:
    for term in terms['id']:
        if term in row:
            label = True
            break
        else:
            label = False
    is_apoptosis.append(label)
df['apoptosis_related'] = is_apoptosis

df.to_csv("data_labeled.csv")