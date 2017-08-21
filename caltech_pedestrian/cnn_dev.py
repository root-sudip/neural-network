import sys
import os
import operator as op

import numpy as np

from PIL import Image

import pandas as pd
import csv

from keras.models import Sequential
from keras.layers import Dense, Activation

from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_json


import readline
from os import listdir

class cnn_dev:
	def __init__(self,no_epoch=None):
		print('CNN object initialized ...')
		if no_epoch == None:
			self.no_epoch = 150 # by default no of epoch 150
		else:
			self.no_epoch = no_epoch

	def load_data(self,p_path_name,n_path_name,samples=None):
		self.training_data_list = []
		j = 0
		if samples == None:
			print('Enter number of samples')
		else:
			self.Y_train_d = np.zeros([samples])

		#for positive training sample
		p = 0
		for p_file_name in sorted(os.listdir(p_path_name)):
			print("\rPositive File Name : ",p_file_name,end="")
			self.img_array = np.asarray(Image.open(p_path_name+p_file_name).resize((32,32), Image.ANTIALIAS))
			self.training_data_list.append(self.img_array)
			self.Y_train_d[j] = 1
			j = j + 1
			p = p + 1

		print()
		print('Positive training set loading completed.')
		#for negative training sample
		n = 0 
		for n_file_name in sorted(os.listdir(n_path_name)):
			print("\rNegative File Name : ",n_file_name,end="")
			self.img_array = np.asarray(Image.open(n_path_name+n_file_name).resize((32,32), Image.ANTIALIAS))
			self.training_data_list.append(self.img_array)
			self.Y_train_d[j] = 0
			j = j + 1
			n = n + 1

		print()
		print('Negative training set loading completed.')


		self.array_list_l = np.asarray(self.training_data_list)
		self.X_train = np.reshape(self.array_list_l,(samples,32,32,3))
		self.Y_train = np_utils.to_categorical(self.Y_train_d)

		print('Total number of Positive Sample : ',p)
		print('Total number of Negative Sample : ',n)

		print('X_train shape : ',self.X_train.shape)
		print('Y_train one hot shape : ',self.Y_train.shape)

	def create_model(self):
		self.model = Sequential()
		
		self.model.add(Conv2D(512,(5,5),padding='same',input_shape=self.X_train.shape[1:]))
		self.model.add(Activation('sigmoid'))
		self.model.add(MaxPooling2D(pool_size=(3,3),strides=None))

		self.model.add(Flatten())
		self.model.add(Dense(512))
		self.model.add(Activation('sigmoid'))

		self.model.add(Dense(2))
		self.model.add(Activation('softmax'))

		self.model.compile(loss='categorical_crossentropy',optimizer='sgd',matrix=['accuracy'])

	def train_model(self):
		self.best_accuracy = 0.0
		for i in range(0,self.no_epoch):
			print('Iteration == ',i)
			self.accuracy = self.model.fit(self.X_train, self.Y_train, nb_epoch=1,batch_size = 120)
			print(self.accuracy.history.keys())
			self.iter_accuracy = op.itemgetter(0)(self.accuracy.history['acc'])
			if (self.best_accuracy < self.iter_accuracy):
				self.best_accuracy = self.iter_accuracy
			self.save_model()
		print('After Interation best accuracy is : ',self.best_accuracy)

	def save_model(self):
		model_json = self.model.to_json()
		with open("model_cnn.json", "w") as json_file:
			json_file.write(model_json)
		self.model.save_weights("model_cnn.h5")

	def load_model(self):
		json_file = open('model_cnn.json', 'r')
		self.model = json_file.read()
		json_file.close()
		self.model = model_from_json(self.model)
		self.model.load_weights("model_cnn.h5")

	def test_model(self):
		pass

ob = cnn_dev()

if sys.argv[1] == 'train':
	ob.load_data(p_path_name=sys.argv[2],n_path_name=sys.argv[3],samples=248617)
	ob.create_model()
	ob.train_model()
	ob.save_model()

elif sys.argv[1] == 'test':
	pass

elif sys.argv[1] == '':
	print('You should use train/test.')

else:
	print('You should write the argv parameters.')
