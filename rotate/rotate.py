"""
Developer: Sudip Das
Licence : Indian Statistical Institute
"""

from PIL import Image
import os

path_list = ['set0/','set1/']


for dir_name in path_list:
	for file in os.listdir(dir_name):
		if file.endswith('.png'):
			img = Image.open(dir_name+file).rotate(120,expand=True)
			img.save('rotate/'+dir_name+'120/'+file)
			print(file)