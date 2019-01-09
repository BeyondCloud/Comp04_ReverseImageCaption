import json
import os
from os.path import join, isfile
import re
import numpy as np
import pickle
import argparse
import skipthoughts
import h5py

import time
import pandas as pd
# from tqdm import tqdm

data_path = '../compDir/dataset/'
df = pd.read_pickle(data_path+'/text2ImgData.pkl')
num_training_sample = len(df)
n_images_train = num_training_sample
print('There are %d image in training data'%(n_images_train))
df.head(5)
df['ImagePath'].values[:2]
train_lst = []
for img_path in df['ImagePath'].values:
    train_lst.append(img_path[-15:])

data_dir = "Data"
img_dir = join(data_dir, 'flowers/jpg')

image_files = [f for f in os.listdir(img_dir) if 'jpg' in f]
# print(image_files[300:400])  #['image_03157.jpg', 'image_06464.jpg'....
# print(len(image_files)) #8189
image_captions = { img_file : [] for img_file in image_files }

caption_dir = join(data_dir, 'flowers/text_c10')
class_dirs = []
for i in range(1, 103):
    class_dir_name = 'class_%.5d'%(i)
    class_dirs.append( join(caption_dir, class_dir_name))

 
for class_dir in class_dirs:
    caption_files = [f for f in os.listdir(class_dir) if 'txt' in f]
    
    for cap_file in caption_files:
        with open(join(class_dir,cap_file)) as f:
            captions = f.read().split('\n')
        img_file = cap_file[0:11] + ".jpg"
#         print(img_file)
        # 5 captions per image (max 11)
        image_captions[img_file] += [cap for cap in captions if len(cap) > 0][0:5]

model = skipthoughts.load_model()
encoded_captions = {}


for i, img in enumerate(train_lst):
    st = time.time()
    encoded_captions[img] = skipthoughts.encode(model, image_captions[img])
    print(i, len(train_lst), img)
    print("Seconds", time.time() - st)


h = h5py.File(join(data_dir, 'flower_train.h5'))
for key in encoded_captions:
    h.create_dataset(key, data=encoded_captions[key])
h.close()
