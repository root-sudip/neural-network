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
	# print(result[i])

	image = cv2.imread(i)
	print('>',i)
	for j in result[i]:
		print(j)
		cv2.rectangle(image, (int(float(j[1])), int(float(j[2]))), (int(float(j[3]))+int(float(j[1])), int(float(j[4]))+int(float(j[2]))), (255,0,0), 2)
	
	img_s = Image.fromarray(image)
	img_s.save("dump2/"+i.strip('dataset/train2014/'))
	k = k + 1