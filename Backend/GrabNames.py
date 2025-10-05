import numpy as np
import pandas as pd
import pprint as pp
#########################

# Reading in the CSV file

fp = file_path = "/Users/randytauyan/Documents/GitHub/Space-Brigade/Datasets(Temp)/sbdb_query_results.csv"
df = pd.read_csv(fp, sep=';')
# Showing the file columns
print(df.head())
NamesId = [(name,id) for name in df['full_name'] for id in df['spkid']]
pp.pprint(NamesId[100])