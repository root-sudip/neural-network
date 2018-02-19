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
import os.path

from shutil import copyfile

file_name = sys.argv[1]


result = {}

with open(file_name) as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in csvreader:
        #print(row)
        if row[0] in result:
            result[row[0]].append(row[1:6])
        else:
            result[row[0]] = [row[1:6]]

# for i in result:
# 	#print(i)
# 	print(result[i])

# print(result)



k = 0
for i in sorted(result):

	path = 'dump/'+str(i)+'/'
	#print('>',i)
	for j in result[i]:
		print(j[0])

		if not os.path.isfile('dump/'+str(i)+'/'+j[0].strip('dataset/train2014/')):
			copyfile(j[0], path+j[0].strip('dataset/train2014/'))

		print(i,',',path+j[0].strip('dataset/train2014/'),',',j[1],',',j[2],',',j[3],',',j[4])

		