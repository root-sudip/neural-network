import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense, Activation
import pandas as pd
import csv
from sklearn.datasets import load_iris
from keras.utils import np_utils
import operator as op

from keras.datasets import mnist

class mlp:
	def __init__(self,no_epoch):
		print('mlp ceated...')
		self.no_epoch = no_epoch

	def load_data(self):

		# self.iris = load_iris()
		# self.X_train = self.iris.data # features data
		# self.Y_train = self.iris.target #target data
		# self.column_names = self.iris.feature_names
		(self.X_train, self.Y_train), (self.X_test, self.Y_test) = mnist.load_data()
		self.X_train = self.X_train.reshape(60000, 784)
		self.X_test = self.X_test.reshape(10000, 784)
		self.input_dim = self.X_train[1].shape
		print('X_train shape : ',self.X_train.shape)
		print('X_test shape : ',self.X_test.shape)
		self.X_train = self.X_train.astype('float32')
		self.X_test = self.X_test.astype('float32')
		self.X_train /= 255
		self.X_test /= 255
		
	def create_model(self):
		self.Y_train = np_utils.to_categorical(self.Y_train)
		self.model = Sequential()
		self.model.add(Dense(output_dim=512, input_dim=784))
		self.model.add(Activation("relu"))
		self.model.add(Dense(output_dim=10, input_dim=10))
		self.model.add(Activation("relu"))
		self.model.add(Dense(output_dim=10))
		self.model.add(Activation("softmax"))
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


	def save_model(self):
		model_json = self.model.to_json()
		with open("model_cnn.json", "w") as json_file:
			json_file.write(model_json)
		self.model.save_weights("model_cnn.h5")

	def load_mode(self):
		json_file = open('mod.json', 'r')
		self.model = json_file.read()
		json_file.close()
		self.model = model_from_json(self.model)
		self.model.load_weights("model_16_net.h5")



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

ob.load_data()
ob.create_model()
ob.train_model()
ob.test_model()