import pandas as pd

df = pd.read_csv("/home/ubuntu/DeepIO/switch/data/csvs/SW02-04-05-06/03 July 2022 11_12_29.csv", 
                 on_bad_lines='skip', 
                 low_memory=False)

print(df.head())
print("\n")
print(df.columns)
print("\n")
print(df.info())