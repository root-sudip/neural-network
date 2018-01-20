"""python3 new_interface.py resized/ out/"""



import os
import numpy as np
import sys
import glob
from PIL import Image

path = sys.argv[1]


for dirName, subdirList, fileList in os.walk(sys.argv[1]):
	for match in fileList:
		print(match)
		if match.lower()[-4:] in ('.jpg', '.png', '.gif', 'jpeg'):
			im = Image.open(path+match).resize((623,623), Image.ANTIALIAS)
			im.save(sys.argv[2]+match)