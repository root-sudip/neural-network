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

import os
from PIL import Image

class mlp:
	def __init__(self,no_epoch):
		print('mlp ceated...')
		self.no_epoch = no_epoch

	def load_data(self,path_name,samples):

		# self.iris = load_iris()
		# self.X_train = self.iris.data # features data
		# self.Y_train = self.iris.target #target data
		# self.column_names = self.iris.feature_names
		self.all_array_list = []
		self.Y_train = np.zeros([8734])
		j = 0 
		for dirName, subdirList, fileList in os.walk(path_name):
			i = 0 
			for subDir in subdirList:
				print('sub dir : ',subDir)
				for fname in os.listdir(dirName+'/'+subDir+'/'):
					print('File Name : ',fname)
					self.img_array = np.asarray(Image.open(dirName+'/'+subDir+'/'+fname).resize((32,32), Image.ANTIALIAS))
					print(self.img_array.shape)
					if self.img_array.ndim == 3:
						self.all_array_list.append(self.img_array)
						self.Y_train[j] = i	
						j = j + 1
				i = i + 1
		print('len : ',j)
		self.array_list_l = np.asarray(self.all_array_list)

		self.X_train = np.reshape(self.array_list_l,(8734,32,32,3))
		print('one_hot : ',self.Y_train)
		print('features_array : ',self.X_train.shape)



	def create_model(self):
		self.Y_train = np_utils.to_categorical(self.Y_train)
		self.model = Sequential()
		self.model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=self.X_train.shape[1:]))
		self.model.add(Activation('relu'))
		self.model.add(Conv2D(32, (3, 3)))
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.25))

		self.model.add(Conv2D(64, (3, 3), padding='same'))
		self.model.add(Activation('relu'))
		self.model.add(Conv2D(64, (3, 3)))
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.25))

		self.model.add(Flatten())
		self.model.add(Dense(512))
		self.model.add(Activation('relu'))
		self.model.add(Dropout(0.5))
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
		print('After Interation best accuracy is : ',self.best_accuracy)


	def test_model(self):
		# self.iris = load_iris()
		# self.X_test = self.iris.data[:2,:] # features data
		# self.Y_test = self.iris.target[:2]


		self.classes = self.model.predict_classes(self.X_test, batch_size=120)

		#get accuration
		self.test_dim = self.Y_test.shape
		print('Test dimention : ',self.test_dim)
		self.accuration = np.sum(self.classes == self.Y_test)/self.test_dim * 100

		print ('Test Accuration : ',str(self.accuration),'%')
		print ('Prediction :',self.classes)
		print ('Target :',np.asarray(self.Y_test,dtype="int32"))

ob = mlp(50)

ob.load_data('101_ObjectCategories',300)
ob.create_model()
ob.train_model()
#ob.test_model()