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



class mlp:
	def __init__(self,no_epoch):
		print('mlp ceated...')
		self.no_epoch = no_epoch

	def load_data(self,path_name,samples):

		self.all_array_list = []
		self.Y_train = np.zeros([2121])
		j = 0 
		for dirName, subdirList, fileList in os.walk(path_name):
			i = 0 
			for subDir in subdirList:
				
				num_of_image = 0 
				for fname in os.listdir(dirName+'/'+subDir+'/'):
					#print('File Name : ',fname)
					self.img_array = np.asarray(Image.open(dirName+'/'+subDir+'/'+fname).resize((32,32), Image.ANTIALIAS))
					# print(self.img_array.shape)
					if self.img_array.ndim == 3:
						self.all_array_list.append(self.img_array)
						self.Y_train[j] = i	
						j = j + 1
						num_of_image = num_of_image + 1
						if num_of_image < 40:
							num_of_image = num_of_image + 1
						else: 
							break
				print('sub dir : => ',subDir,'Number of Image => ',num_of_image,'label => ',i)
				i = i + 1
		self.array_list_l = np.asarray(self.all_array_list)
		print('Total number of image => ',j)
		self.X_train = np.reshape(self.array_list_l,(2121,32,32,3))
		print('one_hot : ',self.Y_train)
		print('features_array : ',self.X_train.shape)




	def create_model(self):
		self.Y_train = np_utils.to_categorical(self.Y_train)
		self.model = Sequential()
		self.model.add(Conv2D(10, (10, 10), padding='same',
                 input_shape=self.X_train.shape[1:]))
	
		self.model.add(Activation('sigmoid'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
	

		self.model.add(Conv2D(12, (10, 10), padding='same'))
		self.model.add(Activation('sigmoid'))
	
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
	


		self.model.add(Conv2D(14, (10, 10), padding='same'))
		self.model.add(Activation('sigmoid'))
	
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		


		self.model.add(Flatten())
		self.model.add(Dense(102))
		
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

	def test_model(self,filename,no_samples):
	
		self.test_list = []
		self.X_test = np.asarray(Image.open(filename).resize((32,32), Image.ANTIALIAS))
		self.test_list.append(self.X_test)
		self.X_test = np.asarray(self.test_list)

		self.Y_test = np.zeros([no_samples])
		self.Y_test[0] = 11
	
		self.classes = self.model.predict_classes(self.X_test, batch_size=120)


		self.test_dim = self.Y_test.shape
		print('Test dimention : ',self.test_dim)
		self.accuration = np.sum(self.classes == self.Y_test)/1 * 100

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

ob = mlp(50)

if sys.argv[1] == 'test':
	print('Trying to predict ...')
	ob.load_model()
	ob.test_model('image_0012.jpg',1)
elif sys.argv[1] == 'train_model':
	ob.load_data('101_ObjectCategories',300)
	ob.create_model()
	ob.train_model()
	ob.save_model()
else:
	print('You should write the argv parameter.')