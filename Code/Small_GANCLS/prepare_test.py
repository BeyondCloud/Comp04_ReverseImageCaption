import json
import os
from os.path import join, isfile
import re
import numpy as np
import pickle
import argparse
import skipthoughts
import h5py

import pandas as pd
# from tqdm import tqdm

data_path = './Data'
df = pd.read_csv(data_path+'/test.csv',dtype={'ID':'str'})
num_testing_sample = len(df)
n_images_test = num_testing_sample
print('There are %d image in testing data'%(n_images_test))
print(df.head(5))

model = skipthoughts.load_model()
encoded_captions = {}


for i, img in enumerate(df['ID'].values):
    encoded_captions[img] = skipthoughts.encode(model, [df['Captions'].values[i]])
    print(i, n_images_test, img)


h = h5py.File('./Data/flower_test.h5')
for key in encoded_captions:
    h.create_dataset(key, data=encoded_captions[key])
h.close()
