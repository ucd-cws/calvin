import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('storage.csv', index_col=0, parse_dates=True)
df = df.filter(like='SR_')
# print df.min(axis=0) / df.max(axis=0)
df.describe().transpose().to_csv('SR_stats.csv')
