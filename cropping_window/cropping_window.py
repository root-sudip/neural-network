"""
Developer: Sudip Das.
Licence: Indian Statistical Institute.

$python3 cropping window.py input_dir output_dir
"""
from PIL import Image
import os
import numpy as np
import sys

win_size = (100,100)
height = 1280
width = 720
k = 0



for file in os.listdir(sys.argv[1]):
	print(file)
	img = np.asarray(Image.open(sys.argv[1]+file))
	for i in range(0,height,win_size[0]):
		for j in range(0,width,win_size[1]):
			try:
				crop = img[i:i+win_size[0],j:j+win_size[1]]
				im = Image.fromarray(crop)
				im.save(sys.argv[2]+"image_"+str(k)+".png")
				k = k + 1
			except ValueError:
				pass


print('Total number of Images : ',k)
