from PIL import Image
import sys
import os

import numpy as np

import shutil

count = 0


if sys.argv[1] == 'trim_size':

	for file in sorted(os.listdir(sys.argv[2])):
		img = np.asarray(Image.open(sys.argv[2]+file))

		if img.shape[1] > 10 and img.shape[1] < 50:
			count = count + 1
			shutil.copy2(sys.argv[2]+file, sys.argv[3])
			print("\rshape : ",img.shape,end="")

	print()
	print('Total number of sorted images : ',count)

elif sys.argv[1] == 'max_width':
	max_width = 0
	for file in sorted(os.listdir(sys.argv[2])):
		img = np.asarray(Image.open(sys.argv[2]+file))

		if max_width < img.shape[1]:
			max_width = img.shape[1]
			print("\rFile Name : ",file,end="")
	print()
	print('Max width : ',max_width)


elif sys.argv[1] == 'padding':

	c = 0
	for file in sorted(os.listdir(sys.argv[2])):
		old_im = Image.open(sys.argv[2]+file)
		old_size = old_im.size

		w = int((170*old_size[0])/49)

		old_image_resized = Image.open(sys.argv[2]+file).resize((w,200), Image.ANTIALIAS)
		old_image_resized_size = old_image_resized.size

		new_size = (256, 256)
		new_im = Image.new("RGB", new_size)
		new_im.paste(old_image_resized, (int((new_size[0]-old_image_resized_size[0])/2),int((new_size[1]-old_image_resized_size[1])/2)))
		new_im.save(sys.argv[3]+str(c)+".png")
		c = c + 1

elif sys.argv[1] == '':
	print('You should use train/test.')

else:
	print('You should write the argv parameter.')