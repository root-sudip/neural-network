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
import glob

path = sys.argv[1]




result = {}

for match in glob.iglob("%s/*/*" % path,recursive=True):
	if match.lower()[-4:] in ('.csv'):
		with open(match) as csvfile:
			csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
			for row in csvreader:
        		#print(row)
				if row[0] in result:
					result[row[0]].append(row[1:5])
				else:
					result[row[0]] = [row[1:5]]

k = 0
for i in sorted(result):
	
	split = i.split('/')
	str_split = split[1].split('_')

	set_name = str_split[0]+str_split[1]
	vide0_name = str_split[2]+'_'+str_split[3]

	print(split[1])

	image = Image.open(set_name+'/'+vide0_name+'/'+split[1])
	for j in result[i]:
		#pass
		img2 = image.crop((int(j[0]), int(j[1]), int(j[2]), int(j[3])))
		img2.save("training/positive/"+str(k)+'.png')
		k = k + 1


print('Total number of pedestrian : ',k)

