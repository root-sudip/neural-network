"""
Developer: Sudip Das
Licence : Indian Statistical Institute
"""

import csv
import sys
from PIL import Image
import cv2
import csv

file_name = sys.argv[1]




result = {}

with open(file_name) as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in csvreader:
        #print(row)
        if row[0] in result:
            result[row[0]].append(row[1:5])
        else:
            result[row[0]] = [row[1:5]]

# for i in result:
# 	print(i)
k = 0
for i in sorted(result):
	print(i.strip('video_00/'))
	image = cv2.imread(i)
	for j in result[i]:
		cv2.rectangle(image, (int(j[0]), int(j[1])), (int(j[2]), int(j[3])), (255,0,0), 2)
		#print(j[0],"," ,j[1],"," ,j[2],"," ,j[3])
		#print(">",j)
	img_s = Image.fromarray(image)
	img_s.save("dump1/"+i.strip('video_00/'))
	k = k + 1