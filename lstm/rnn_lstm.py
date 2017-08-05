from __future__ import print_function
import numpy as np
import pandas

import pandas as pd
import csv
#from sklearn.datasets import load_iris

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
from keras.models import model_from_json
from keras.utils import np_utils
import operator as op

import os
from PIL import Image

import sys

import readline
from os import listdir
import random


class rnn:
	def __init__(self,no_epoch):
		print('......')
		self.no_epoch = no_epoch

		self.path = "got.txt"

		try: 
			self.text = open(self.path).read().lower()
		except UnicodeDecodeError:
			import codecs
			self.text = codecs.open(path, encoding='utf-8').read().lower()

		print('corpus length:', len(self.text))

		#self.chars = set(self.text)
		self.words = set(open('got.txt').read().lower().split())

		# self.array = np.asarray

		#print("chars:",type(self.chars))
		print("words",type(self.words))
		print("total number of unique words",len(self.words))
		#print("total number of unique chars", len(self.chars))


		self.word_indices = dict((c, i) for i, c in enumerate(self.words))
		self.indices_word = dict((i, c) for i, c in enumerate(self.words))


		print("word_indices", type(self.word_indices), "length:",len(self.word_indices) )
		print("indices_words", type(self.indices_word), "length", len(self.indices_word))

		self.maxlen = 4
		self.step = 1
		print("maxlen:",self.maxlen,"step:", self.step)
		self.sentences = []
		self.next_words = []
		self.next_words= []
		self.sentences1 = []
		self.list_words = []

		self.sentences2=[]
		self.list_words=self.text.lower().split()#collecting the list of words
		
		# print(self.list_words[0+self.maxlen])
		for i in range(0,len(self.list_words)-self.maxlen, self.step):
			self.sentences2 = ' '.join(self.list_words[i: i + self.maxlen])
			#print('^',self.sentences2,' : ' ,i+self.maxlen)#
			self.sentences.append(self.sentences2)
			self.next_words.append((self.list_words[i + self.maxlen]))
			#print('=',self.list_words[i + self.maxlen])#
		print('nb sequences(length of sentences):', len(self.sentences))
		print("length of next_word",len(self.next_words))
		#for 

		self.X_T = []
		self.X = np.zeros((self.maxlen,len(self.word_indices)))
		self.y = np.zeros((len(self.sentences),len(self.word_indices)))
		#print(self.word_indices)
		j = 0
		for sentence in self.sentences:
			i = 0
			for word in sentence.split():
				#print(word)
				#print(self.word_indices[word])
				#label = self.word_indices
				#print(self.word_indices[word])
				self.X[i][self.word_indices[word]] = 1
				#print(self.X[i,:])
				i = i + 1
			self.X_T.append(self.X)
			self.X.fill(0)
			#print(self.next_words[i])
			self.y[j][self.word_indices[self.next_words[j]]] = 1
			j = j + 1
		#print('& ',self.word_indices[0])
		
		#self.y_train = np_utils.to_categorical(self.y)
		#print('shape of Y train : ',self.y_train.shape)
		self.X_tt = np.asarray(self.X_T)
		#print('before x ',self.X_tt.shape)
		self.X_train = np.reshape(self.X_tt,(len(self.sentences),4,len(self.word_indices)))
		#self.X_train = np_utils.to_categorical(self.X_tt)
		print('X train shape : ',self.X_train.shape)
		print('Y train shape : ',self.y.shape)		


	def create_model(self):
		self.model = Sequential()
		self.model.add(LSTM(512, return_sequences=True, input_shape=(self.maxlen, len(self.word_indices))))
		self.model.add(Dropout(0.2))
		self.model.add(LSTM(512, return_sequences=False))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(512))
		self.model.add(Dense(self.y.shape[1]))
		#model.add(Dense(1000))
		self.model.add(Activation('softmax'))

		self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])
	
		
	def train_model(self):

		self.best_accuracy = 0.0
		for iteration in range(0, self.no_epoch):
			print('Iteration == ',iteration)
			self.accuracy_measures = self.model.fit(self.X_train, self.y, batch_size=128, epochs=1)
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

	def test_model(self,filename=None,no_samples=None,label=None):

		start_index = random.randint(0, len(self.list_words) - self.maxlen - 1)
		sentence = self.list_words[start_index: start_index + self.maxlen]
		print('Sequence : ',sentence)
		X_test = np.zeros((self.maxlen, len(self.word_indices)))
		
		i = 0
		for word in sentence:
			X_test[i][self.word_indices[word]] = 1												
			i = i + 1
		X_test = np.reshape(X_test, (1, self.maxlen, len(self.word_indices)))
		#X_test = np_utils.to_categorical(self.X_test)
		print('X_test shape : ',X_test.shape)
		# for t, word in enumerate(sentence):
		# 	X_test[0, t, self.word_indices[word]] = 1.
		preds = self.model.predict(X_test, verbose=0)
		print('Confidence value : ',preds)
		next_index = np.argmax(preds)
		#next_index = sample(preds, diversity)
		next_word = self.indices_word[next_index]
		#generated += next_word
		#generated += ' '.join(sentence)
		print('generated text : ',next_word)

def completer(text, state):
	options = [x for x in listdir('.') if x.startswith(text)]
	try:
		return options[state]
	except IndexError:
		return None


readline.set_completer(completer)
readline.parse_and_bind("tab: complete")

ob = rnn(150)

if sys.argv[1] == 'test':
	print('Trying to predict ...')
	ob.load_model()
	ob.test_model()
elif sys.argv[1] == 'train':
	#ob.load_data('101_ObjectCategories',samples=7496)
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
