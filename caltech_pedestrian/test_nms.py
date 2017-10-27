def test_model(self,filename,frame_size, strides):

		coordinates = []

		image = cv.imread(filename)
		(winW, winH) = (128, 400)
		k = 0
		color_change = 0
		for resized in pyramid(image, scale=0):
		# loop over the sliding window for each layer of the pyramid
			for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
			# if the window does not meet our desired window size, ignore it
				if window.shape[0] != winH or window.shape[1] != winW:
					continue
 
				croped_image =  np.reshape(np.asarray(Image.fromarray(window).resize((32,32), Image.ANTIALIAS)),(1,32,32,3))
				classes = self.model.predict_classes(croped_image, batch_size=1)

				if classes == 1:
					prob = self.model.predict(croped_image)[0]
					print('confidence value : ',prob[1])
					if prob[1] > .70:
						cv.rectangle(image, (x, y), (x + winW, y + winH), (255, 0, 0), 1)
						coordinates.append([x,y,x+winW,y+winH])
				else:
					#cv.rectangle(img, (i, j), (i + frame_size[0], j + frame_size[1]), (0, 0, 255), 1)
					pass

				save_img = Image.fromarray(window)
				save_img.save("croped/"+str(k)+".png")
				k = k + 1

		img_save = Image.fromarray(image)
		img_save.save("resized_img.png")

		# print('Coordinates : ',coordinates)
		as_array_coordinates = np.array(coordinates)
		print('Coordinates : ',as_array_coordinates)

		images = [(sys.argv[2],np.array(coordinates))]

		#print('Final : ',images)

		for (imagePath, boundingBoxes) in images:
		# load the image and clone it
			print ('[x] %d initial bounding boxes' % (len(boundingBoxes)))
			image = cv.imread(imagePath)
			orig = image.copy()
 
		# loop over the bounding boxes for each image and draw them
		for (startX, startY, endX, endY) in boundingBoxes:
			cv.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)
 
		# perform non-maximum suppression on the bounding boxes
		pick = non_max_suppression_slow(boundingBoxes, 0.3)
		print ('[x] after applying non-maximum, %d bounding boxes' % (len(pick)))
 
		# loop over the picked bounding boxes and draw them
		for (startX, startY, endX, endY) in pick:
			cv.rectangle(image, (startX, startY), (endX, endY), (0, 255, color_change), 2)
			color_change = color_change + 255


		img_save = Image.fromarray(image)
		img_save.save("resized_img_nms.png")