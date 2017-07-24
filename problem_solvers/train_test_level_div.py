
import numpy as np
import os
import glob
from PIL import Image
class div_train_test:

	def __init__(self,train_label=None,test_label=None):
		self.train_label = train_label
		self.test_label = test_label

	def count_file(self,suddir_name):
		return len(glob.glob1(suddir_name,"*.jpg"))
		# return 

	def load_data(self,path_name,p_of_training_samples=None,p_of_test_sample=None):

			if p_of_training_samples == None and p_of_test_sample == None:
				self.p_of_training_samples = 80
				self.p_of_test_sample = 20
			else:
				self.p_of_training_samples = p_of_training_samples
				self.p_of_test_sample = p_of_test_sample
			with open(self.train_label,"a") as trainl, open(self.test_label,"a") as testl:
				i  = 0 
				for dirName, subdirList, fileList in os.walk(path_name):
					# fl.write('sub dir => %s Number og Images => %d label => %d \n' % (subDir,len(subDir,i))
					i = 0					
					for dir_name in subdirList:
						train_count = 0
						test_count = 0
						total_train_test = 0
						if self.count_file(path_name+dir_name) < 70:
							self.training_sample = int((self.count_file(path_name+dir_name) * self.p_of_training_samples)/100)
							# print('traing sample',self.training_sample)
							self.test_sample = int((self.count_file(path_name+dir_name) * self.p_of_test_sample)/100)
							print('dir name =>',dir_name,'train samples => ',self.test_sample,'test samples => ',self.training_sample)
						else:
							self.training_sample = int((70 * self.p_of_training_samples)/100)
							self.test_sample = int((70 * self.p_of_test_sample)/100)
							print('dir name =>',dir_name,'train samples => ',self.test_sample,'test samples => ',self.training_sample)
						for file_name in os.listdir(path_name+dir_name):
							if total_train_test < (self.training_sample+self.test_sample):
								self.img_array = np.asarray(Image.open(dirName+'/'+dir_name+'/'+file_name).resize((32,32), Image.ANTIALIAS))
								if train_count < self.training_sample and self.img_array.ndim == 3:
									trainl.write(' %s,%s,%d\n' % (dir_name,file_name,i))
									train_count = train_count + 1
									#subdir,filename,label
								elif self.img_array.ndim == 3:
									testl.write(' %s,%s,%d\n' % (dir_name,file_name,i))
									test_count = test_count + 1
								total_train_test = total_train_test + 1
							else:
								break
						i = i + 1
				trainl.close()
				testl.close()
ob = div_train_test('train_label.txt','test_label.txt')
ob.load_data('101_ObjectCategories/',)