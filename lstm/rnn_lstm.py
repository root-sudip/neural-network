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
		

		for i in range(0,len(self.list_words)-self.maxlen, self.step):
			self.sentences2 = ' '.join(self.list_words[i: i + self.maxlen])
			#print('^',self.sentences2,' : ' ,i+self.maxlen)#
			self.sentences.append(self.sentences2)
			self.next_words.append((self.list_words[i + self.maxlen]))

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
				self.X[i][self.word_indices[word]] = 1
				i = i + 1
			self.X_T.append(self.X)
			self.X.fill(0)
			self.y[j][self.word_indices[self.next_words[j]]] = 1
			j = j + 1

		self.X_tt = np.asarray(self.X_T)
		self.X_train = np.reshape(self.X_tt,(len(self.sentences),4,len(self.word_indices)))

		print('X train shape : ',self.X_train.shape)
		print('Y train shape : ',self.y.shape)		


	def create_model(self):
		self.model = Sequential()
		self.model.add(LSTM(512, return_sequences=True, input_shape=(self.maxlen, len(self.word_indices))))
		#self.model.add(Dropout(0.2))
		# self.model.add(LSTM(70, return_sequences=True))
		# self.model.add(LSTM(100,return_sequences=True))
		#self.model.add(Dropout(0.2))
		self.model.add(Dense(512))
		self.model.add(Activation('sigmoid'))
		#self.model.add(Dropout(0.2))
		self.model.add(LSTM(512, return_sequences=False))
		#self.model.add(Dropout(0.2))
		#self.model.add(Dense(512))
		self.model.add(Dense(self.y.shape[1]))
		#model.add(Dense(1000))
		self.model.add(Activation('softmax'))

		self.model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=["accuracy"])
	
		
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

		print('X_test shape : ',X_test.shape)

		preds = self.model.predict(X_test, verbose=0)[0]
		#print('Confidence value : ',preds)
		next_index = np.argmax(preds)
		next_word = self.indices_word[next_index]
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
	ob.create_model()
	ob.train_model()
	ob.save_model()
else:
	print('You should write the argv parameters.')


