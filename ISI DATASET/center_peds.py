"""
Developer: Sudip Das
Licence : Indian Statistical Institute
"""

import csv
import sys
from PIL import Image
import cv2
import csv
import numpy as np
import sys

import glob
import os 
import transform_cordinates as tc

result = {}
path = sys.argv[1]
for dirName, subdirList, fileList in os.walk(path):
	for subdir_name in subdirList:
		for match in os.listdir(dirName+'/'+subdir_name+'/'):
			if match.lower()[-4:] in ('.csv'):
				with open(dirName+'/'+subdir_name+'/'+match) as csvfile:
					csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
					for row in csvreader:
        				#print(row)
						if row[0] in result:
							result[row[0]].append(row[1:5])
						else:
							result[row[0]] = [row[1:5]]

with open(sys.argv[2],"a") as file:

	k = 0
	for i in sorted(result):
		# image = cv2.imread(i)
		split = i.split('/')
		str_split = split[1].split('_')

		set_name = str_split[0]+str_split[1]
		vide0_name = str_split[2]+'_'+str_split[3]

		print(split[1])
		image = Image.open(set_name+'/'+vide0_name+'/'+split[1])
		#image1 = np.asarray(image)
		for j in result[i]:

			yy,yy_ = tc.transform(int(j[1]),int(j[3]))

			x = int(j[0])
			y = int(yy)

			x_ = int(j[2]) - int(j[0])
			y_ = yy_ - yy
			x_ = x_/2
			y_ = y_/2
			x = x + x_
			y = y + y_


			file.write(i)
			file.write(',')
			file.write(str(x))
			file.write(',')
			file.write(str(y))
			file.write(',')
			file.write(str(j[0]))
			file.write(',')
			file.write(str(yy))
			file.write(',')
			file.write(str(j[2]))
			file.write(',')
			file.write(str(yy_))
			file.write('\n')
			img2 = image.crop((int(j[0]), int(j[1]), int(j[2]), int(j[3])))
			#img2.save("training/positive/"+str(k)+'.png')
			k = k + 1


print('Total number of pedestrian : ',k)



