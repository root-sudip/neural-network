"""
command to run: $python3 cnn_dev.py train training/ v_label.csv

Copyright (C) 2017 Sudip Das <d.sudip47@gmail.com>.
Licence : Indian Statistical Institute

"""

import sys
import os
import operator as op

import numpy as np

from PIL import Image

#import pandas as pd
import csv

from keras.models import Sequential
from keras.layers import Dense, Activation

from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_json


from PIL import ImageFont, ImageDraw, ImageEnhance
import cv2 as cv

import readline
from os import listdir

from random import shuffle



class cnn_dev:
	def __init__(self,no_epoch=None):
		print('CNN object initialized ...')
		if no_epoch == None:
			self.no_epoch = 10 # by default no of epoch 150
		else:
			self.no_epoch = no_epoch


	def intersection_over_union(self,boxA, boxB):
		# determine the (x, y)-coordinates of the intersection rectangle
		xA = max(boxA[0], boxB[0])
		yA = max(boxA[1], boxB[1])
		xB = min(boxA[2], boxB[2])
		yB = min(boxA[3], boxB[3])
 
		# compute the area of intersection rectangle
		interArea = (xB - xA + 1) * (yB - yA + 1)
 
		# compute the area of both the prediction and ground-truth
		# rectangles
		boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
		boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
		# compute the intersection over union by taking the intersection
		# area and dividing it by the sum of prediction + ground-truth
		# areas - the interesection area
		iou = interArea / float(boxAArea + boxBArea - interArea)
 
		# return the intersection over union value
		return iou

	def label(self,filename=None):
		print('Labeling ... ')
		file = open('label.csv','a')

		print()
		for dirName, subdirList, fileList in os.walk(sys.argv[2]): 
			l = 0
			for subdir_name in subdirList:
				for fname in os.listdir(dirName+'/'+subdir_name+'/'):
					print("\033[44m"+"\rFile Name : ",fname,"\033[0m",end="")
					file.write(dirName)
					file.write(subdir_name)
					file.write("/")
					file.write(fname)
					file.write(",")
					file.write(str(l))
					file.write("\n")
				l = l + 1
			print()
		file.close()



	def load_path(self):

		self.list_csv_training = []
		with open("label.csv","r") as fl:
			j = 0 
			csv_reader = csv.reader(fl)
			print('Loading Path ....')
			for row in csv_reader:
				print("\033[44m"+"\rFile Name : ",row[0],"  Label : ",row[1],"\033[0m",end="")
				self.list_csv_training.append(str(row))
				j = j + 1
			print()

		shuffle(self.list_csv_training)
		print('Path Loading Completed.')


	def load_data(self,samples_start=None,samples_end=None,samples=None):
		
		if samples == None:
			print('Enter number of samples')
		else:
			Y_train_d = np.zeros([samples])

			training_data_list = []
			i = 0
			j = 0
			for path in self.list_csv_training:

				if samples_start<= i <samples_end:

					temp = path.split(',')
					#print('Path : ',temp[0].replace('[','').replace("'",''),'Label :',temp[1].replace(']','').replace("'",''))
					#print()

					print("\033[44m"+"\rFile Name : ",temp[0].replace('[','').replace("'",""),"  Label : ",temp[1].replace(']','').replace("'",""),"\033[0m",end="")
					# print()
					img_array = np.asarray(Image.open(temp[0].replace('[','').replace("'",'')).resize((32,32), Image.ANTIALIAS))
					training_data_list.append(img_array)
					Y_train_d[j] = int(temp[1].replace(']','').replace("'",""))
					i = i + 1
					j = j + 1
				else:
					i = i + 1

			print()
			print('Total number of file read : ',j)

			array_list_l = np.asarray(training_data_list)
			X_train = np.reshape(array_list_l,(samples,32,32,3))
			Y_train = np_utils.to_categorical(Y_train_d)

			return X_train, Y_train

	def load_data_for_validation(self,samples=None):
		if samples == None:
			print('Enter number of samples')
		else:
			validation_data_list = []
			Y_validation_d = np.zeros([samples])
			print('Loading validation data')

			j = 0
			fl = open('v_label.csv','r')
			csv_reader = csv.reader(fl)

			for row in csv_reader:

					print("\033[44m"+"\rFile Name : ",row[0],"  Label : ",row[1],"\033[0m",end="")
					img_array = np.asarray(Image.open(row[0]).resize((32,32), Image.ANTIALIAS))
					validation_data_list.append(img_array)
					Y_validation_d[j] = int(row[1])
					j = j + 1
			print()
			print('Total number of file read : ',j)

			array_list_l = np.asarray(validation_data_list)
			print(array_list_l.shape)
			self.X_validation = np.reshape(array_list_l,(samples,32,32,3))
			self.Y_validation = np_utils.to_categorical(Y_validation_d)



	def create_model(self):
		self.model = Sequential()
		
		self.model.add(Conv2D(10,(5,5),padding='same', strides=4, input_shape=(32,32,3)))
		self.model.add(Activation('sigmoid'))
		self.model.add(MaxPooling2D(pool_size=(4,4),strides=2))

		#self.model.add(Dropout(0.25))

		self.model.add(Conv2D(7,(5,5),padding='same',strides=2))
		self.model.add(Activation('sigmoid'))
		self.model.add(MaxPooling2D(pool_size=(2,2),strides=2))

		#self.model.add(Dropout(0.25))

		self.model.add(Conv2D(10,(5,5),padding='same',strides=1))
		self.model.add(Activation('sigmoid'))
		#self.model.add(MaxPooling2D(pool_size=(2,2),strides=1))


		#self.model.add(Conv2D(10,(7,7),padding='same',strides=1))
		#self.model.add(Activation('sigmoid'))

		# self.model.add(Conv2D(15,(7,7),padding='same',strides=1))
		# self.model.add(Activation('sigmoid'))


		self.model.add(Dense(20))

		#self.model.add(Dropout(0.25))

		self.model.add(Flatten())


		self.model.add(Dense(2))
		self.model.add(Activation('softmax'))

		self.model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

	def train_model(self, samples=None,train_sample=None):

		if samples == None:
			print('Enter number of samples for training set')
		elif train_sample == None:
			print('Enter of sample in each block')
		else:
			best_accuracy = 0.0
			best_val_accuracy = 0.0

			temp = samples

			iteration = 0

			for i in range(0,self.no_epoch):
				print('Iteration == ',i)
				start = 0
				t_accuracy = 0
				count = 0
				avg_accuracy = 0
				for j in range(0,samples):
					
					if samples < 0:
						break
					else:
						end = start + train_sample

						#need a condition to sorting out if in case number of samples are few in a block

						if samples < train_sample:
							samples_size = samples
							samples = samples - train_sample
							print('Sample : ',samples_size)
						else:
							samples_size = train_sample
							samples = samples - train_sample

						#end

						X_train, Y_train = self.load_data(samples_start=start, samples_end=end, samples=samples_size)
						#,validation_data = (self.X_validation, self.Y_validation), verbose = 1
						accuracy = self.model.fit(X_train, Y_train, epochs=1,batch_size = 100,verbose = 1)
						print(accuracy.history.keys())

						iter_accuracy = op.itemgetter(0)(accuracy.history['acc'])

						t_accuracy = t_accuracy + iter_accuracy

						#iter_val_accuracy = op.itemgetter(0)(accuracy.history['val_acc'])

						
						start = end
						print('Block :',j,' Number of samples : ',samples_size)
						count = count + 1
				self.save_model()

				avg_accuracy = t_accuracy/count

				print('\033[92m'+'Avg Accuracy : ',avg_accuracy,'\033[0m')

				iteration = iteration + 1

				samples = temp

				score = self.model.evaluate(self.X_validation, self.Y_validation, batch_size=100, verbose=1)

				print()
				
				print('\033[92m'+'Validation loss : ',score[0],'\033[0m')
				print('\033[92m'+'Validation accuracy : ',score[1],'\033[0m')


				if (best_accuracy < avg_accuracy):
					best_accuracy = iter_accuracy

				if (best_val_accuracy < score[1]):
					best_val_accuracy = score[1]
			
			print('After ',iteration,'th iteration best training accuracy is : ',best_accuracy)
			print('After',iteration,'th iteration best validation accuracy is : ',best_val_accuracy)

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

	def test_model(self,filename=None,output_filename=None,frame_size=None, strides=None):

		#.resize((32,32), Image.ANTIALIAS)


		img = np.asarray(Image.open(filename))
		print('Imput image shape for testing : ',img.shape)
		width = img.shape[0]
		height = img.shape[1]
		p = 0
		p_iou = 0

		previous = []
		new = []

		for i in range(0,height,strides[1]):
			for j in range(0,width,strides[0]):
				try:
					#cv.rectangle(img, (i, j), (i + frame_size[0], j + frame_size[1]), (255, 0, 0), 1)
					crop = np.asarray(Image.fromarray(img[i:i+frame_size[0],j:j+frame_size[1]]).resize((32,32), Image.ANTIALIAS))

				

					croped_image = np.reshape(crop,(1,32,32,3))
					# #print('croped image shape : ',croped_image.shape)

					classes = self.model.predict_classes(croped_image, batch_size=1)
				

					if classes == 0:
						#need a condition to reduce the number of boxes

						if p_iou > 0:

							#making the list of coordinates of new box
							new.append(i)
							new.append(j)
							new.append(i + frame_size[0])
							new.append(j + frame_size[1])
							#end

							iou = self.intersection_over_union(previous,new) * 100
							print('IOU : ',iou)

							if iou <= 100.0:
								cv.rectangle(img, (i, j), (i + frame_size[0], j + frame_size[1]), (255, 0, 0), 1)
								previous = new
								new[:] = []
							else:
								pass
								#previous = new
							p_iou = p_iou + 1
							p = p + 1

						else:
							cv.rectangle(img, (i, j), (i + frame_size[0], j + frame_size[1]), (255, 0, 0), 1)

							#making the list of coordinates of previous box
							previous.append(i)
							previous.append(j)
							previous.append(i + frame_size[0])
							previous.append(j + frame_size[1])
							#end

							p_iou = p_iou + 1
							p = p + 1

						#end conditions

					else:
						#cv.rectangle(img, (i, j), (i + frame_size[0], j + frame_size[1]), (0, 0, 255), 1)
						pass
						
					print("\rPediction : ",classes," Total nuber of pedestrain : ",p ,end="")
					
				except ValueError:
					#print('except')
					pass
		print()
		#print('Total number of pedestrian : ',p)

		out = Image.fromarray(img)

		if output_filename == None:
			out.save("rectangle.png")
		else:
			out.save(output_filename)

ob = cnn_dev(20)

if sys.argv[1] == 'train':
	ob.load_path()
	ob.load_data_for_validation(samples=52349)
	ob.create_model()
	ob.train_model(samples=291566,train_sample=5000)
	ob.save_model()

elif sys.argv[1] == 'test':
	ob.load_model()
	ob.test_model(filename=sys.argv[2],output_filename=sys.argv[3],frame_size=(22,42),strides=(30,10))

elif sys.argv[1] == '':
	print('You should use train/test.')

elif sys.argv[1] == 'label':
	ob.label()

else:
	print('You should write the argv parameters.')
