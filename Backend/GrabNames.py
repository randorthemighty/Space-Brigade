import numpy as np
import pandas as pd
import pprint as pp
#########################

# Reading in the CSV file

## What We Want to Do is use this file without having to use local files
fp = file_path = "Datasets(Temp)/Sentry Earth Impact Monitoring.xlsx"
df = pd.read_csv(fp, sep=';')
# Showing the file columns
print(df.head())
NamesId = [(name,id) for name in df['full_name'] for id in df['spkid']]
pp.pprint(NamesId[100])