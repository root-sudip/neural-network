from skimage.feature import hog
import sys
import os

from PIL import Image

import numpy as np
class hog:
	def __init__(self,path):
		self.training_path = path
	def load_data(self):
		for dirnamelist,subdirlist,filelist in os.walk(self.training_path):
			for filename in filelist:
				im = np.asarray(Image.open(self.training_path+filename).resize((32,32), Image.ANTIALIAS))
				hd =  hog(im, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),visualise=True)
				#print(hd)


obj = hog(sys.argv[1])
obj.load_data()