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
			self.text = open(self.path).read().lower().replace(',','').replace('"','')
			print(self.text)
		except UnicodeDecodeError:
			import codecs
			self.text = codecs.open(path, encoding='utf-8').read().lower().replace(',','').replace('"','')
			print(self.text)
		#self.text = self.textt.strip(',')

		print('corpus length:', len(self.text))

		#self.chars = set(self.text)
		
		self.words = set(open('got.txt').read().replace(',','').replace('"','').lower().split())

		# self.array = np.asarray

		#print("chars:",type(self.chars))
		print("words",type(self.words))
		print("total number of unique words",len(self.words))
		# #print("total number of unique chars", len(self.chars))


		self.word_len = len(self.words)
		self.maxlen = 4
		self.step = 1

		#print(self.word_len)
		# print("word_len", type(self.word_len), "length:",len(self.word_len) )
		# print("indices_words", type(self.indices_word), "length", len(self.indices_word))
		# with open("lstm_label.csv","a") as file:
		# 	l = 0 
		# 	for i in self.words:
		# 		file.write(str(l))
		# 		file.write(',')
		# 		file.write(i)
		# 		file.write('\n')
		# 		l = l + 1
		# 	file.close()

	def load_data(self):
		print('Loading data ...')
		self.maxlen = 4
		self.step = 1
		print("maxlen:",self.maxlen,"step:", self.step)
		self.sentences = []
		self.next_words = []
		#self.next_words= []
		#self.sentences1 = []
		self.list_words = []#list of word from corpus

		# self.sentences2 = []
		self.list_words = self.text.lower().split()#collecting the list of words
		

		for i in range(0,len(self.list_words)-self.maxlen, self.step):
			#print(self.list_words[i:i+self.maxlen])
			self.sentences.append(' '.join(self.list_words[i:i+self.maxlen]))
			#self.sentences2 = ' '.join(self.list_words[i: i + self.maxlen])
			#print('^',self.sentences2,' : ' ,i+self.maxlen)#
			#self.sentences.append(self.sentences2)
			#self.next_words.append((self.list_words[i + self.maxlen]))
			#print('Printing the list of sentances ...')''.join
			#print(self.sentences)
			self.next_words.append(self.list_words[i+self.maxlen])

		print('nb sequences(length of sentences):', len(self.sentences))
		print("length of next_word",len(self.next_words))

		self.X_T = []
		self.X = np.zeros((self.maxlen,self.word_len))
		self.y = np.zeros((len(self.sentences),self.word_len))
		#print(self.word_len)
		j = 0
		f = open("ro.txt","a")
		for sentence in self.sentences:
			print(sentence)
			i = 0
			f.write('{')
			f.write('\n')
			f.write('Input :')
			f.write('\n')
			
			for word in sentence.split():
				#print(word)
				with open("lstm_label.csv") as fd1:
					csv_reader = csv.reader(fd1)
					for ww in csv_reader:
						#print(ww[1],':',word)
						if ww[1] == word:
							print('^',ww[1])
							l_word1 = int(ww[0])
							print(l_word1)
							self.X[i][l_word1] = 1
							break
					fd1.close()

				f.write(str(self.X[i,:]))
				f.write('\n')
				f.write(word)
				f.write('\n')
				#print(self.X[i,:])
				i = i + 1
			self.X_T.append(self.X)
			#print(': ',self.X)
			self.X.fill(0)
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
		self.X_tt = np.asarray(self.X_T)
		self.X_train = np.reshape(self.X_tt,(len(self.sentences),4,self.word_len))

		print('X train shape : ',self.X_train.shape)
		print('Y train shape : ',self.y.shape)		


	def create_model(self):
		print('Initializing model ...')
		self.model = Sequential()
		self.model.add(LSTM(64, return_sequences=True, input_shape=(self.maxlen, self.word_len)))

		self.model.add(LSTM(128, return_sequences=False))
		# self.model.add(Dropout(0.2))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(self.y.shape[1]))
		self.model.add(Activation('softmax'))

		self.model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=["accuracy"])
		
	def train_model(self):
		print('Training model .... ')
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
		print('Model save ... ')
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

	def test_model(self,filename=None,no_samples=None,label=None):

		start_index = random.randint(0, len(self.list_words) - self.maxlen - 1)
		sentence = self.list_words[start_index: start_index + self.maxlen]
		print('Sequence : ',sentence)
		X_test = np.zeros((self.maxlen, self.word_len))	
		i = 0
		for word in sentence:
			#print(word)
			with open("lstm_label.csv") as fd3:
				csv_reader = csv.reader(fd3)
				for ww in csv_reader:
					if ww[1] == word:
						print(ww[1],' : ',word)
						l_word3 =int(ww[0])
						break
				fd3.close()
			print(l_word3)
			X_test[i][l_word3] = 1	
			#print(X_test[i,:])											
			i = i + 1
		X_test = np.reshape(X_test, (1, self.maxlen, self.word_len))
		#np.savetxt('matrix.txt',X_test,fmt="%d")
		print('X_test shape : ',X_test.shape)

		preds = self.model.predict(X_test, verbose=0)
		#print(self.word_len)cat ls
		next_index = np.argmax(preds)
		print('Confidence value : ',next_index)
		word2l = ''
		with open("lstm_label.csv") as fd4:
			csv_reader = csv.reader(fd4)
			for ww in csv_reader:
				if ww[0] == next_index:
					word2l =ww[1]
					break
			fd4.close()	
		#next_word = self.indices_word[word2l]
		print('generated text : ',word2l)

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
else:
	print('You should write the argv parameters.')


