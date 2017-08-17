"""
Developer: Sudip Das
Licence : Indian Statistical Institute
"""

from __future__ import print_function
import numpy as np
import pandas
import pandas as pd

import csv

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

		self.no_epoch = no_epoch
		self.path = "got.txt"

		try: 
			self.text = open(self.path).read().lower().replace(',','').replace('"','')
		except UnicodeDecodeError:
			import codecs
			self.text = codecs.open(path, encoding='utf-8').read().lower().replace(',','').replace('"','')


		print('corpus length:', len(self.text))
		
		self.words = set(open(self.path).read().replace(',','').replace('"','').lower().split())

		print('Text : ',self.text)

		print("words",type(self.words))
		print("total number of unique words",len(self.words))


		self.word_len = len(self.words)
		self.maxlen = 1 #max len
		self.step = 1 #stride size


		print("maxlen:",self.maxlen,"step:", self.step)
		self.sentences = [] #list of samples for training
		self.next_words = [] #list of targeted words for training

		self.list_words = [] #list of words from corpus
		self.list_words = self.text.lower().split()#storing the list of words
		
		for i in range(0,len(self.list_words)-self.maxlen, self.step):
			self.sentences.append(' '.join(self.list_words[i:i+self.maxlen]))
			self.next_words.append(self.list_words[i+self.maxlen])

		print('nb sequences(length of sentences):', len(self.sentences))
		print("length of next_word",len(self.next_words))

	def  make_label(self): # writing the csv file for labeling
		self.words = set(open(self.path).read().replace(',','').replace('"','').lower().split())
		with open("lstm_label.csv","a") as file:
			l = 0 
			for i in self.words:
				file.write(str(l))
				file.write(',')
				file.write(i)
				file.write('\n')
				l = l + 1
			file.close()

	def load_data(self):
		print('Loading data ...')

		self.X = np.zeros((len(self.sentences),self.maxlen,self.word_len))#making the training samples array
		self.y = np.zeros((len(self.sentences),self.word_len))#making the target array for samples
		j = 0
		f = open("ro.txt","a") # for log
		k = 0
		for sentence in self.sentences:
			i = 0
			f.write('{')
			f.write('\n')
			f.write('Input :')
			f.write('\n')
			
			for word in sentence.split():
				with open("lstm_label.csv") as fd1:
					csv_reader = csv.reader(fd1)
					for ww in csv_reader:
						if ww[1] == word:
							l_word1 = int(ww[0])
							self.X[k,i,l_word1] = 1
							break
					fd1.close()

				f.write(str(self.X[k,i,:]))
				f.write('\n')
				f.write(word)
				f.write('\n')

				i = i + 1
			k = k + 1

			f.write('Target :')
			
			with open("lstm_label.csv") as fd2:
				csv_reader = csv.reader(fd2)
				for ww in csv_reader:
					if ww[1] == self.next_words[j]:
						#
						l_word2 =int(ww[0])
						self.y[j][l_word2] = 1
						break
			
				f.write(str(self.y[j,:]))
				f.write('\n')
				f.write(self.next_words[j])
				f.write('\n')
				
				fd2.close()	
			f.write('}')
			j = j + 1

		f.close()

		print('X train shape : ',self.X.shape)
		print('Y train shape : ',self.y.shape)		


	def create_model(self):
		print('Initializing model ...')
		self.model = Sequential()

		self.model.add(LSTM(256, return_sequences=True, input_shape=(self.maxlen, self.word_len)))
		self.model.add(LSTM(512, return_sequences=False))

		self.model.add(Dense(self.y.shape[1]))
		self.model.add(Activation('softmax'))

		self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])
		
	def train_model(self):
		print('Training model .... ')
		self.best_accuracy = 0.0
		for iteration in range(0, self.no_epoch):
			print('Iteration == ',iteration)
			self.accuracy_measures = self.model.fit(self.X, self.y, batch_size=128, epochs=1)
			print(self.accuracy_measures.history.keys())
			self.iter_accuracy = op.itemgetter(0)(self.accuracy_measures.history['acc'])
			if (self.best_accuracy < self.iter_accuracy):
				self.best_accuracy = self.iter_accuracy
			self.save_model()
		print('After',self.no_epoch,'Interation best accuracy is : ',self.best_accuracy)	

	def save_model(self):
		print('Model saved ... ')
		model_json = self.model.to_json()
		with open("model_cnn.json", "w") as json_file:
			json_file.write(model_json)
		self.model.save_weights("model_cnn.h5")

	def load_model(self):
		print('Loading model ... ')
		json_file = open('model_cnn.json', 'r')
		self.model = json_file.read()
		json_file.close()
		self.model = model_from_json(self.model)
		self.model.load_weights("model_cnn.h5")

	def test_model(self):

		word = input('Enter word : ')
		
		X_test = np.zeros((self.maxlen, self.word_len))
		X_test = np.reshape(X_test, (1, self.maxlen, self.word_len))	
		predict_word = ''
		sentence = ''
		while True:

			X_test.fill(0)
			if word.endswith(('.','?')):
				break
			else:
				print()
				with open("lstm_label.csv") as fd3:
			 		csv_reader = csv.reader(fd3)
			 		for ww in csv_reader:
			 			if ww[1] == word:
			 				l_word3 = int(ww[0])
			 				X_test[0,0,l_word3] = 1
			 				break
			 		fd3.close()
				preds = self.model.predict(X_test, verbose=0)
				print('Prdict shape : ',preds.shape)
				next_index1 = preds.flatten()
				most_three = next_index1.argsort()[-3:][::-1]
				next_index = most_three[0] #np.argmax(preds)

				with open("lstm_label.csv") as fd4:
					csv_reader = csv.reader(fd4)
					for ww in csv_reader:
						if ww[0] == str(next_index):
							word2l = ww[1]
							word = str(word2l)
						
							if word in sentence.split():
								for ww in csv_reader:
									next_index = most_three[1]
									if ww[0] == str(next_index):
										word2l = ww[1]
										word = str(word2l)
										print('Generated/Predicted text : ',word2l)
										sentence = sentence + str(word2l) + ' '
										break
							else:
								print('Generated/Predicted text : ',word2l)
								sentence = sentence + str(word2l) + ' '

					fd4.close()

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
	ob.load_data()
	ob.create_model()
	ob.train_model()
	ob.save_model()
elif sys.argv[1] == 'label':
	ob.make_label()
elif sys.argv[1] == '':
	print('You should write the argv parameters.')
else:
	print('You should use train/test/label as argv argument')