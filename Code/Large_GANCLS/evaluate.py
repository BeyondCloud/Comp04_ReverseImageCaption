# python ./inception_score.py ../foo/inference/ ./score/foo.csv

import numpy as np
import tensorflow as tf
from model import *
from tqdm import tqdm
import pandas as pd
import os
import scipy
from scipy.io import loadmat
import re
import string
from utils import *
import random
import time
import argparse
import nltk
import warnings
from PIL import ImageEnhance,Image
warnings.filterwarnings('ignore')

dictionary_path = './dictionary'


checkpoint_dir = './checkpoint'

z_dim = 512         # Noise dimension
c_dim = 3           # for rgb
batch_size = 64
ni = int(np.ceil(np.sqrt(batch_size)))
img_size = 64
t_dim = 256
gf_dim = 128


###################################
test_sentence = pd.read_csv('test.csv',dtype={'ID':'str'})


def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))



t_real_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='real_caption_input')
fake_image = tf.placeholder('float32', [batch_size, img_size, img_size, 3], name = 'fake_image')


t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')
vocab = np.load(dictionary_path+'/vocab.npy')

model_options = {
    'z_dim' : z_dim,
    'batch_size' : batch_size,
    'img_size' :img_size,
    't_dim' :t_dim,
    'gf_dim' :gf_dim ,
    'vocab_size':len(vocab)
}

net_g = Generator(t_z, TextEncoder(t_real_caption,option = model_options , is_training=False, reuse=False).out,option = model_options
                ,is_training=False, reuse=False)

disc = Discriminator(fake_image, TextEncoder(t_real_caption,option = model_options , is_training=False, reuse=True).out, is_training=False, reuse=False)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

# ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
saver = tf.train.Saver()

try:
    # loader = tf.train.Saver(var_list=tf.global_variables())
    # load(loader, sess, checkpoint_dir+'/VBN700/modelVBN.ckpt')


    ckpt_path  = checkpoint_dir+'/VBN700/modelVBN.ckpt'
    saver.restore(sess, ckpt_path)
    print("Restore model",ckpt_path)
    # load(loader, sess, checkpoint_dir+'/backup/my_model.ckpt')
    
except:
    raise('no checkpoints found.')


n_caption_test = len(test_sentence['Captions'])
n_batch_epoch = int(n_caption_test / batch_size) + 1

print(n_batch_epoch)

regen = 10
for i in tqdm(range(n_batch_epoch-1)):
    # test_cap = caption[i*batch_size: (i+1)*batch_size]
    all_cap = []
    for j in range(batch_size):
        test_cap = test_sentence['Captions'].values[i*batch_size+j]
        all_cap.append(sent2IdList(test_cap))

    best_imgs = np.zeros((batch_size,64,64,3))
    best_score = np.ones((batch_size,))
    for j in range(regen):
        z = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, z_dim)).astype(np.float32)
        gen = sess.run(net_g.out, feed_dict={t_real_caption: all_cap, t_z: z})
        score = sess.run(disc.out, feed_dict={fake_image: gen,t_real_caption: all_cap}).reshape([-1])
        for k in range(batch_size):
            if score[k]<best_score[k]:
                best_imgs[k,:,:,:] = gen[k,:,:,:]
                best_score[k] = score[k]

    for j in range(batch_size):
        image_path = 'inference/inference_'+test_sentence['ID'].values[i*batch_size+j]+'.png'
        images =  best_imgs[j]*0.5+0.5

        #########################################
        img = Image.fromarray(np.uint8(images*255))
        enhancer = ImageEnhance.Sharpness(img)
        img  = enhancer.enhance(3.0)


        #don't use color enhance
        # enhancer = ImageEnhance.Color(img)
        # images  = np.array(enhancer.enhance(1.3))

        #########################################

        scipy.misc.imsave(image_path,(img))
################################### deal with last batch

all_cap = []
for i in range(n_caption_test-batch_size,n_caption_test):
    test_cap = test_sentence['Captions'].values[i]
    all_cap.append(sent2IdList(test_cap))
z = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, z_dim)).astype(np.float32)
gen = sess.run(net_g.out, feed_dict={t_real_caption: all_cap, t_z: z})

for i in range(batch_size-(n_caption_test%batch_size),batch_size):
    image_path = 'inference/inference_'+test_sentence['ID'].values[n_caption_test-batch_size+i]+'.png'
    images =  gen[i]*0.5+0.5

    #########################################
    img = Image.fromarray(np.uint8(images*255))
    enhancer = ImageEnhance.Sharpness(img)
    img  = enhancer.enhance(3.0)

    #don't use color enhance
    # enhancer = ImageEnhance.Color(img)
    # images  = np.array(enhancer.enhance(1.3))

    #########################################

    scipy.misc.imsave(image_path,(img))

