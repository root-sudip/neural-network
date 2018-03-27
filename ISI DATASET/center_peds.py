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

# for i in result:
# 	print(i)

with open(sys.argv[2],"a") as file:

	k = 0
	for i in sorted(result):
		print(i)
		# image = cv2.imread(i)
		image = Image.open(i)
		#image1 = np.asarray(image)
		for j in result[i]:
			#img2 = image.crop((int(j[0]), int(j[1]), int(j[2]), int(j[3])))

			w, h = image.size

			x = w/2
			y = h/2

			file.write(i)
			file.write(',')
			file.write(str(x))
			file.write(',')
			file.write(str(y))
			file.write('\n')

			#img2.save("dump/"+str(k)+'.png')
			k = k + 1

		#cv2.rectangle(image, (int(j[0]), int(j[1])), (int(j[2]), int(j[3])), (255,0,0), 2)
		#print(j[0],"," ,j[1],"," ,j[2],"," ,j[3])
		#print(">",j)
	#img_s = Image.fromarray(image)

print('Total number of pedestrian : ',k)


