# python ./inception_score.py ../master_text-to-image/Data/inference/ ./score/test.csv

import tensorflow as tf
import numpy as np
import model
import pickle
from os.path import join
import h5py
from Utils import image_processing
import scipy.misc
import random
import json
import os

def main():

	# model_path = './Data/Models/ep_10_nice1152.ckpt'

	# model_path = './Data/Models/ep_195_nice.ckpt'

	model_path = './Data/Models/temp.ckpt'

	epochs = 200
	data_dir = "Data"
	image_size = 64
	z_dim = 100
	caption_vector_length = 4800
	n_images = 1
	caption_thought_vectors = 'Data/flower_test.h5'
	model_options = {
	    'z_dim' : z_dim,
	    't_dim' :256,
	    'batch_size' : n_images,
	    'image_size' :64,
	    'gf_dim' : 64,
	    'df_dim' : 64,
	    'gfc_dim' :1024,
	    'caption_vector_length' : caption_vector_length
	}

	gan = model.GAN(model_options)
	_, _, _, _, _,_ = gan.build_model()
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.InteractiveSession(config = config)

	saver = tf.train.Saver()
	saver.restore(sess, model_path)
	
	input_tensors, outputs = gan.build_generator()

	h = h5py.File( caption_thought_vectors )

	# caption_vectors = np.array(h['vectors'])
	caption_image_dic = {}

	for ds in h.items():

		caption_image_dic[ds[0]] = np.array(ds[1])

	result = {}
	for cn in caption_image_dic:
		caption_images = []
		z_noise = np.random.uniform(-1, -0.9, [n_images, z_dim])
		# z_noise = np.random.normal(0, 0.1,  [n_images, z_dim])
		caption_vector = caption_image_dic[cn][0]
		caption = [ caption_vector[:caption_vector_length] ] * n_images
		
		[ gen_image ] = sess.run( [ outputs['generator'] ], 
			feed_dict = {input_tensors['t_real_caption'] : caption,input_tensors['t_z'] : z_noise})
		
		caption_images = [gen_image[i,:,:,:] for i in range(0, n_images)]
		result[ cn ] = caption_images
		print("Generated", cn)

	for f in os.listdir( join(data_dir, 'val_samples')):
		if os.path.isfile(f):
			os.unlink(join(data_dir, 'val_samples/' + f))

	for cn in result:
		caption_images = []
		for im in  result[ cn ]:
			# im_name = "caption_{}_{}.jpg".format(cn, i)
			# scipy.misc.imsave( join(data_dir, 'val_samples/{}'.format(im_name)) , im)
			caption_images.append( im )
			caption_images.append( np.zeros((64, 5, 3)) )
		combined_image = np.concatenate( caption_images[0:-1], axis = 1 )
		# combined_image = rnd_flip(combined_image)
		scipy.misc.imsave( join(data_dir, 'inference/inference_{}.png'.format(cn)) , combined_image)


if __name__ == '__main__':
	main()
