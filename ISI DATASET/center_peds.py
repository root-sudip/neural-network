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


result = {}
path = sys.argv[1]

for match in glob.glob("%s/*" % path):
	if match.lower()[-4:] in ('.csv'):
		with open(match) as csvfile:
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
		print(i)
		# image = cv2.imread(i)
		image = Image.open(i)
		#image1 = np.asarray(image)
		for j in result[i]:

			x = int(j[0])
			y = int(j[1])

			x_ = int(j[2]) - int(j[0])
			y_ = int(j[3]) - int(j[1])
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
			file.write(str(j[1]))
			file.write(',')
			file.write(str(j[2]))
			file.write(',')
			file.write(str(j[3]))
			file.write('\n')

			#img2.save("dump/"+str(k)+'.png')
			k = k + 1


print('Total number of pedestrian : ',k)


