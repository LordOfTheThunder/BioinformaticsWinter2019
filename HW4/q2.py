import pandas as pd
import re

df = pd.read_csv("DE_results_corrected.csv")
df = df[['row', 'baseMean',	'log2FoldChange', 'lfcSE', 'stat', 'pvalue', 'padj']]
#df = pd.read_csv("data\lung_counts.csv")

df = df.dropna(subset=['padj'])
df = df.sort_values(['padj'])

df = df.reset_index()

find = re.compile(r"^[^.]*")

for i in range(0, len(df.index)):
    df.at[i, 'row'] = re.search(find, df.iloc[i]['row']).group(0)
   #df.at[i, 'ensgene'] = re.search(find, df.iloc[i]['ensgene']).group(0)

df['row'].to_csv("ranked_genes.txt", index=False, header=False)
# df['row'].to_csv("target_list.txt", index=False, header=False)
#df['row'].to_csv("background_list_.txt", index=False, header=False)