	def test_model(self,filename,frame_size, strides):

		image = cv.imread(filename)
		(winW, winH) = (128, 128)
		k = 0
		for resized in pyramid(image, scale=0):
		# loop over the sliding window for each layer of the pyramid
			for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
			# if the window does not meet our desired window size, ignore it
				if window.shape[0] != winH or window.shape[1] != winW:
					continue
 
				# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
				# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
				# WINDOW
 
				# since we do not have a classifier, we'll just draw the window
				#clone = resized.copy()
				croped_image =  np.reshape(np.asarray(Image.fromarray(window).resize((32,32), Image.ANTIALIAS)),(1,32,32,3))
				classes = self.model.predict_classes(croped_image, batch_size=1)
				if classes == 1:
					print('confidence value : ',classes[1])
					cv.rectangle(image, (x, y), (x + winW, y + winH), (255, 0, 0), 1)
				else:
					#cv.rectangle(img, (i, j), (i + frame_size[0], j + frame_size[1]), (0, 0, 255), 1)
					pass

				save_img = Image.fromarray(window)
				save_img.save("croped/"+str(k)+".png")
				k = k + 1


		img_save = Image.fromarray(image)
		img_save.save("resized_img.png")