import pandas as pd
from data_reader import read_consumption_and_weather

_, _, df = read_consumption_and_weather()

df1 = df['NO1']
df2 = df['NO2']
df3 = df['NO3']
df4 = df['NO4']
df5 = df['NO5']

print(df1.describe())