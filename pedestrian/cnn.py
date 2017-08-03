import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense, Activation
import pandas as pd
import csv
#from sklearn.datasets import load_iris
from keras.utils import np_utils
import operator as op

from keras.datasets import mnist

from keras.layers import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.models import model_from_json

import os
from PIL import Image

import sys

import readline
from os import listdir

import csv

class mlp:
	def __init__(self,no_epoch):
		print('cnn ceated...')
		self.no_epoch = no_epoch

	def load_data(self,p_path_name=None,n_path_name=None,p_samples=None,n_samples=None):
		print('Loading data from : ',path_name)
		self.list_of_images = []
		self.Y_train = np.zeros([samples])

		#for positive features
		i = 0
		for dirnameList, subdirList, filenameList in os.walk(p_path_name):
			for filename in filenameList:
				im = np.asarray(Image.open(path_name+filename).resize((32,32), Image.ANTIALIAS))
				self.list_of_images.append(im)
				self.Y_train[i] = 0
				i = i + 1

		#for negative features
		for dirnameList, subdirList, filenameList in os.walk(n_path_name):
			for filename in filenameList:
				im = np.asarray(Image.open(path_name+filename).resize((32,32), Image.ANTIALIAS))
				self.list_of_images.append(im)
				self.Y_train[i] = 1
				i = i + 1

		self.X_train = np.asarray(self.list_of_images)
		self.X_train = np.reshape(self.X_train,((p_path_name+n_path_name),32,32,3))
		

	def create_model(self):
		self.Y_train = np_utils.to_categorical(self.Y_train)
		print('one_hot : ',self.Y_train)

		self.model = Sequential()
		self.model.add(Conv2D(512, (5, 5), padding='same',
                 input_shape=self.X_train.shape[1:]))
	
		self.model.add(Activation('sigmoid'))
		self.model.add(MaxPooling2D(pool_size=(3, 3)))


		#self.model.add(Conv2D(512, (3, 3), padding='same'))
		#self.model.add(Activation('sigmoid'))
		#self.model.add(MaxPooling2D(pool_size=(2, 2)))

		# self.model.add(Conv2D(256, (8, 8), padding='same'))
		# self.model.add(Activation('sigmoid'))
		# self.model.add(MaxPooling2D(pool_size=(16, 16)))


		# self.model.add(Conv2D(256, (8, 2), paddi8g='same'))
		# self.model.add(Activation('sigmoid'))
		# self.model.add(MaxPooling2D(pool_size=(16, 16)))
		
		self.model.add(Flatten())

		self.model.add(Dense(512))
		self.model.add(Activation('sigmoid'))
		
		
		
		self.model.add(Dense(100))
		
		self.model.add(Activation('softmax'))

		self.model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
		
	def train_model(self):
		self.best_accuracy = 0.0
		for i in range(0,self.no_epoch):
			print('Iteration == ',i)
			self.accuracy_measures = self.model.fit(self.X_train, self.Y_train, nb_epoch=1, batch_size=120)
			print(self.accuracy_measures.history.keys())
			self.iter_accuracy = op.itemgetter(0)(self.accuracy_measures.history['acc'])
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

	def test_model(self,filename,no_samples=None,label=None):

		with open("test_label.csv","r") as fl:
			self.test_list = []
			self.Y_test = np.zeros([no_samples])
			j = 0 
			csv_reader = csv.reader(fl)
			
			for row in csv_reader:
				# print('dir name =>',row[0])
				self.X_test = np.asarray(Image.open(path_name+'/'+row[0]+'/'+row[1]).resize((32,32), Image.ANTIALIAS))
				self.test_list.append(self.X_test)
				self.Y_test[j] = row[2]
				print('sub dir : => ',row[0],'label => ',row[2])
				j = j + 1
			fl.close()

		self.test_array_list_l = np.asarray(self.test_list)
		print('Total number of image => ',j)
		self.X_test = np.reshape(self.test_array_list_l,(no_samples,32,32,3))
		print('one_hot : ',self.Y_test)
		#print('features_array : ',self.X_test.shape)
	
		# self.test_list = []
		# self.X_test = np.asarray(Image.open(filename).resize((32,32), Image.ANTIALIAS))
		# self.test_list.append(self.X_test)
		# self.X_test = np.asarray(self.test_list)

		# self.Y_test = np.zeros([no_samples])
		# self.Y_test[0] = label
	
		self.classes = self.model.predict_classes(self.X_test, batch_size=1)


		self.test_dim = self.Y_test.shape
		print('Test dimention : ',self.test_dim)
		self.accuration = np.sum(self.classes == self.Y_test)/no_samples * 100

		print ('Test Accuration : ',str(self.accuration),'%')
		print ('Prediction :',self.classes)
		print ('Target :',np.asarray(self.Y_test,dtype="int32"))


def completer(text, state):
	options = [x for x in listdir('.') if x.startswith(text)]
	try:
		return options[state]
	except IndexError:
		return None


readline.set_completer(completer)
readline.parse_and_bind("tab: complete")

ob = mlp(150)

if sys.argv[1] == 'test':
	print('Trying to predict ...')
	ob.load_model()
	ob.test_model('image_0005.jpg',no_samples=1,label=42)
elif sys.argv[1] == 'train':
	ob.load_data(sys.argv[2],samples=924)
	#ob.create_model()
	#ob.train_model(no_samples=852)
	#ob.save_model()
else:
	print('You should write the argv parameters.')

