import numpy as np
import pandas as pd


# Load the CSV file with a different delimiter (e.g., semicolon)
df = pd.read_csv('wine+quality/winequality-red.csv', delimiter=';')

print(df.head())