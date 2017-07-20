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

	def load_data(self,filename):

		# self.iris = load_iris()
		# self.X_train = self.iris.data # features data
		# self.Y_train = self.iris.target #target data
		# self.column_names = self.iris.feature_names
		(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
		

	def create_model(self):
		self.Y_train = np_utils.to_categorical(self.Y_train)
		self.model = Sequential()
		self.model.add(Dense(output_dim=10, input_dim=4))
		self.model.add(Activation("relu"))
		self.model.add(Dense(output_dim=10, input_dim=10))
		self.model.add(Activation("relu"))
		self.model.add(Dense(output_dim=3))
		self.model.add(Activation("softmax"))
		self.model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
		
	def train_model(self):
		self.best_accuracy = 0.0
		for i in range(0,150):
			print('Iteration == ',i)
			self.accuracy_measures = self.model.fit(self.X_train, self.Y_train, nb_epoch=1, batch_size=120)
			print(self.accuracy_measures.history.keys())
			self.iter_accuracy = op.itemgetter(0)(self.accuracy_measures.history['acc'])
			if (self.best_accuracy < self.iter_accuracy):
				self.best_accuracy = self.iter_accuracy
		print('After Interation best accuracy is : ',self.best_accuracy)


	def test_model(self):
		self.iris = load_iris()
		self.X_test = self.iris.data[:2,:] # features data
		self.Y_test = self.iris.target[:2]


		self.classes = self.model.predict_classes(self.X_test, batch_size=120)

		#get accuration

		self.accuration = np.sum(self.classes == self.Y_test)/3.0 * 100

		print ('Test Accuration : ',str(self.accuration),'%')
		print ('Prediction :',self.classes)
		print ('Target :',np.asarray(self.Y_test,dtype="int32"))

ob = mlp(50)

ob.load_data()
ob.create_model()
ob.train_model()
ob.test_model()