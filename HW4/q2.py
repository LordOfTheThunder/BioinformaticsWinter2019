import pandas as pd
import re

df = pd.read_csv("DE_results_corrected.csv")
df = df[['row', 'baseMean',	'log2FoldChange', 'lfcSE', 'stat', 'pvalue', 'padj']]
#df = pd.read_csv("data\lung_counts.csv")

#df = df.loc[(df['padj'] <= 0.05)]
df = df.loc[(df['log2FoldChange'] > 2) | (df['log2FoldChange'] < -2)]

#df = df.dropna(subset=['padj'])
#df = df.sort_values(['padj'])

df = df.reset_index()

find = re.compile(r"^[^.]*")

for i in range(0, len(df.index)):
    df.at[i, 'row'] = re.search(find, df.iloc[i]['row']).group(0)

print(df)

#df['row'].to_csv("ranked_genes.txt", index=False, header=False)
df['row'].to_csv("target_list_2.txt", index=False, header=False)
#df['row'].to_csv("background_list_.txt", index=False, header=False)

