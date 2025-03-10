import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
#import tensorflow as tf
#import seaborn as sns
#from time import perf_counter
#from sklearn.metrics import classification_report, accuracy_score
#from IPython.display import Markdown, display


# Create a list with the filepaths for training and testing
dir_ = Path('./data/PE/BM/train')
train_filepaths = list(dir_.glob(r'**/*.BMP'))
print(len(train_filepaths))

labels = [str(train_filepaths[i]).split("\\")[-2] for i in range(len(train_filepaths))]
print(labels)

filepath = pd.Series(train_filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')


# Concatenate filepaths and labels
df = pd.concat([filepath, labels], axis=1)
print(df.head())
labelsList=df.Label.unique()
print(labelsList)

# Shuffle the DataFrame and reset index
df = df.sample(frac=1, random_state=0).reset_index(drop=True)
print(df.head())

labelsList=df.Label.unique()
print(labelsList)