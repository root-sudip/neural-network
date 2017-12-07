import os
import sys

from random import shuffle

import csv

from PIL import Image

from keras.utils import np_utils

from random import shuffle

import numpy as np

def load_MNIST(file_name):
    print('Reading data ...')
    X_Train = np.zeros([60000, 784])
    Y_Train = np.zeros([60000])

    i = 0
    k = 0
    with open(file_name) as fl:
        line = csv.reader(fl)
        for row in line:
            if k == 0:
                k = k + 1
                pass
            else:
                for j in range(1,784):
                    X_Train[i][j] = float(row[j])
                Y_Train[i] = row[0]
                i += 1
                k = k + 1

    Y_Train = np_utils.to_categorical(Y_Train)

    X_Train = np.reshape(X_Train, (60000, 28, 28, 1))

    print('X_Train : ',X_Train.shape,' Y_Train : ', Y_Train.shape)

    return X_Train, Y_Train

def load_MNIST_test(file_name):
    print('Reading data ...')
    X_Test = np.zeros([28000, 784])
    Y_Test = np.zeros([28000])

    i = 0
    k = 0
    with open(file_name) as fl:
        line = csv.reader(fl)
        for row in line:
            if k == 0:
                k = k + 1
                pass
            else:
                for j in range(1, 784):
                    X_Test[i][j] = float(row[j])
                Y_Test[i] = row[0]
                i += 1
                k = k + 1

    Y_Test = np_utils.to_categorical(Y_Test)

    X_Test = np.reshape(X_Test, (28000, 28, 28, 1))

    print('X_Test : ', X_Test.shape, ' Y_Test : ', Y_Test.shape)

    return X_Test, Y_Test


def label(filename=None):
	print('Labeling ... ')
	file = open('label.csv','a')
	less_size = 0
	print()
	for dirName, subdirList, fileList in os.walk(sys.argv[2]):
		l = 0
		for subdir_name in subdirList:
			for fname in os.listdir(dirName+'/'+subdir_name+'/'):
				print("\033[44m"+"\rFile Name : ",fname,"\033[0m",end="")
				img = np.asarray(Image.open(dirName+subdir_name+'/'+fname))
				if img.shape[1] < 10:
					less_size = less_size + 1
				else:
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
	print('Total trimed : ',less_size)


def load_path(filename):
    list_csv_training = []
    with open(filename,"r") as fl:
        j = 0
        csv_reader = csv.reader(fl)
        print('Loading data ...')
        for row in csv_reader:
            print("\033[44m"+"\rFilename : ",row[0]," Label : ",row[1],"\033[0m",end="")
            list_csv_training.append(str(row))
            j = j + 1
        print()
        shuffle(list_csv_training)
        print('Path loading completed.')
        return list_csv_training

def load_data(filename, samples=None):
    if samples == None:
        print('Enter number of samples')
    else:
        Y_train_d = np.zeros([samples])

        training_data_list = []
        j = 0

        list_csv_training = load_path(filename)
        print('Loading files ...')
        for path in list_csv_training:
                temp = path.split(',')
                print("\033[44m" + "\rFile Name : ", temp[0].replace('[', '').replace("'", ""), "  Label : ", temp[1].replace(']', '').replace("'", ""), "\033[0m", end="")

                img_array = np.asarray(Image.open(temp[0].replace('[', '').replace("'", '')).resize((28, 28), Image.ANTIALIAS))
                training_data_list.append(img_array)
                Y_train_d[j] = int(temp[1].replace(']', '').replace("'", ""))

                j = j + 1

        print()
        print('Total number of file read : ', j)

        print()
        array_list_l = np.asarray(training_data_list)
        X_train = np.reshape(array_list_l, (samples, 28, 28, 3))
        print('X Train shape : ', X_train.shape)
        #Y_train = np_utils.to_categorical(Y_train_d)
        Y_train = Y_train_d
        print('Y Train shape : ',Y_train.shape)

        return X_train, Y_train


def load_flatten_data(filename,total_samples,features):

    X_Train = np.zeros([total_samples,features])
    Y_Train =  np.zeros([total_samples])

    with open(filename,"r") as fl:
        j = 0
        i = 0
        k = 0
        csv_reader = csv.reader(fl)
        print('Loading data ...')
        for row in csv_reader:
            Y_Train[j] = float(row[0])
            for col in row:
                if i == 0:
                    pass
                else:
                    X_Train[j][k] = float(col)
                    k += 1
                i += 1

            i = 0
            k = 0
            print("\rFeature number %d" % j, end="")
            j += 1
    Y_Train = np_utils.to_categorical(Y_Train)

    print('X Train shape : ', X_Train.shape)
    print('Y Train shape : ', Y_Train.shape)

    return X_Train, Y_Train


#x, y = load_flatten_data('flatten.csv', 287961, 1569)

#X_Train, Y_Train = load_data(filename='label.csv', samples=287961)
