import pandas as pd
path = 'uniprot_raw.tsv'
df = pd.read_csv(path, sep='\t')
print(df.columns.tolist())
print(df.head(3).T)
