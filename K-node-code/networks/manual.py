import pandas as pd
import numpy as np
import os

def init():
  current_path = os.path.dirname(os.path.abspath(__file__))
  df_path = os.path.join(current_path, "data/5Adj_matrix.csv")
  df1_path = os.path.join(current_path, "data/5innate_op.csv")

  # Import Synthetic Network Data
  df = pd.read_csv (df_path, header=None)
  G = np.array(df[df.columns[:]])
  # print(G)
  df1 = pd.read_csv(df1_path, header = None)
  s = np.array(df1[df1.columns[:]])
  # Set n according to the data
  n = len(s[:])
  return G, s, n