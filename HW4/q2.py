import pandas as pd
import re

df = pd.read_csv("DE_results.csv")
#df = pd.read_csv("data\lung_counts.csv")
# df = df.sort_values(['padj'])

find = re.compile(r"^[^.]*")

for i in range(0, len(df.index)):
    df.at[i, 'row'] = re.search(find, df.iloc[i]['row']).group(0)
   # df.at[i, 'ensgene'] = re.search(find, df.iloc[i]['ensgene']).group(0)

print(len(df.index))

df['row'].to_csv("target_list.txt", index=False, header=False)
#df['ensgene'].to_csv("background_list.txt", index=False, header=False)