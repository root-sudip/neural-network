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

# result = {}
# path = sys.argv[1]

# with open(path) as csvfile:
# 	csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
# 	for row in csvreader:
# 		y = int(float(row[2])) - 720
# 		y_= int(float(row[4])) - 1280

# 		with open(sys.argv[2]) as csvreader_recorrect:
# 			csvreader_recorrect.write(row[0])
# 			csvreader_recorrect.write(',')
# 			csvreader_recorrect.write(str(row[1]))
# 			csvreader_recorrect.write(',')
# 			csvreader_recorrect.write(str(y))
# 			csvreader_recorrect.write(',')
# 			csvreader_recorrect.write(str(row[3]))
# 			csvreader_recorrect.write(',')
# 			csvreader_recorrect.write(str(y_))
# 			csvreader_recorrect.write('\n')

def transform(y,y_):
	y = 720 - int(float(y))
	y_= 1280 - int(float(y_))
	return y,y_        				