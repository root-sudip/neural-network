import numpy as np
import pandas

import pandas as pd
import csv
#from sklearn.datasets import load_iris
from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file



from keras.models import model_from_json

import os
from PIL import Image


import sys


import readline
from os import listdir



class mlp:
	def __init__(self,no_epoch):
		print('cnn ceated...')
		self.no_epoch = no_epoch

	def load_data(self,path_name,samples=None):
		path = "got.txt"

		try: 
			text = open(path).read().lower()
		except UnicodeDecodeError:
			import codecs
			text = codecs.open(path, encoding='utf-8').read().lower()

		print('corpus length:', len(text))

		chars = set(text)
		words = set(open('got.txt').read().lower().split())

		print("chars:",type(chars))
		print("words",type(words))
		print("total number of unique words",len(words))
		print("total number of unique chars", len(chars))


		word_indices = dict((c, i) for i, c in enumerate(words))
		indices_word = dict((i, c) for i, c in enumerate(words))

		print("word_indices", type(word_indices), "length:",len(word_indices) )
		print("indices_words", type(indices_word), "length", len(indices_word))

		maxlen = 30
		step = 3
		print("maxlen:",maxlen,"step:", step)
		sentences = []
		next_words = []
		next_words= []
		sentences1 = []
		list_words = []

		sentences2=[]
		list_words=text.lower().split()


		for i in range(0,len(list_words)-maxlen, step):
			sentences2 = ' '.join(list_words[i: i + maxlen])
			sentences.append(sentences2)
			next_words.append((list_words[i + maxlen]))
		print('nb sequences(length of sentences):', len(sentences))
		print("length of next_word",len(next_words))

		print('Vectorization...')
		X = np.zeros((len(sentences), maxlen, len(words)), dtype=np.bool)
		y = np.zeros((len(sentences), len(words)), dtype=np.bool)
		for i, sentence in enumerate(sentences):
    		for t, word in enumerate(sentence.split()):
        		#print(i,t,word)
        		X[i, t, word_indices[word]] = 1
    		y[i, word_indices[next_words[i]]] = 1




	def create_model(self):
		model = Sequential()
		model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(words))))
		model.add(Dropout(0.2))
		model.add(LSTM(512, return_sequences=False))
		model.add(Dropout(0.2))
		model.add(Dense(len(words)))
		#model.add(Dense(1000))
		model.add(Activation('softmax'))

		model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	def sample(a, temperature=1.0):
		# helper function to sample an index from a probability array
		a = np.log(a) / temperature
		a = np.exp(a) / np.sum(np.exp(a))
		return np.argmax(np.random.multinomial(1, a, 1))
		
	def train_model(self):

		self.best_accuracy = 0.0
		for iteration in range(1, 300):
			print('Iteration == ',iteration)
			print()
			print('-' * 50)
			print('Iteration', iteration)
			self.accuracy_measures = model.fit(X, y, batch_size=128, nb_epoch=2)
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
	
		self.test_list = []
		self.X_test = np.asarray(Image.open(filename).resize((32,32), Image.ANTIALIAS))
		self.test_list.append(self.X_test)
		self.X_test = np.asarray(self.test_list)

		self.Y_test = np.zeros([no_samples])
		self.Y_test[0] = label
	
		self.classes = self.model.predict_classes(self.X_test, batch_size=1)


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

ob = mlp(150)

if sys.argv[1] == 'test':
	print('Trying to predict ...')
	ob.load_model()
	ob.test_model('image_0005.jpg',no_samples=1,label=42)
elif sys.argv[1] == 'train_model':
	ob.load_data('101_ObjectCategories',samples=7496)
	ob.create_model()
	ob.train_model()
	ob.save_model()
else:
	print('You should write the argv parameters.')



# sub dir : =>  accordion Number of Image =>  49 label =>  0
# sub dir : =>  butterfly Number of Image =>  86 label =>  1
# sub dir : =>  brain Number of Image =>  84 label =>  2
# sub dir : =>  camera Number of Image =>  47 label =>  3
# sub dir : =>  bass Number of Image =>  50 label =>  4
# sub dir : =>  buddha Number of Image =>  81 label =>  5
# sub dir : =>  beaver Number of Image =>  45 label =>  6
# sub dir : =>  binocular Number of Image =>  32 label =>  7
# sub dir : =>  cellphone Number of Image =>  58 label =>  8
# sub dir : =>  barrel Number of Image =>  45 label =>  9
# sub dir : =>  brontosaurus Number of Image =>  38 label =>  10
# sub dir : =>  cannon Number of Image =>  40 label =>  11
# sub dir : =>  anchor Number of Image =>  41 label =>  12
# sub dir : =>  chair Number of Image =>  61 label =>  13
# sub dir : =>  bonsai Number of Image =>  128 label =>  14
# sub dir : =>  ceiling_fan Number of Image =>  45 label =>  15
# sub dir : =>  ant Number of Image =>  34 label =>  16
