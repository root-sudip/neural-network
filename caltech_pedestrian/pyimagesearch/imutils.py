# import the necessary packages
import numpy as np
import cv2

def translate(image, x, y):
	# define the translation matrix and perform the translation
	M = np.float32([[1, 0, x], [0, 1, y]])
	shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

	# return the translated image
	return shifted

def rotate(image, angle, center=None, scale=1.0):
	# grab the dimensions of the image
	(h, w) = image.shape[:2]

	# if the center is None, initialize it as the center of
	# the image
	if center is None:
		center = (w / 2, h / 2)

	# perform the rotation
	M = cv2.getRotationMatrix2D(center, angle, scale)
	rotated = cv2.warpAffine(image, M, (w, h))

	# return the rotated image
	return rotated

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and
	# grab the image size
	dim = None
	(h, w) = image.shape[:2]

	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (int(w * r), height)

	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))

	# resize the image
	resized = cv2.resize(image, dim, interpolation=inter)

	# return the resized image
	return resized

def skeletonize(image, size, structuring=cv2.MORPH_RECT):
	# determine the area (i.e. total number of pixels in the image),
	# initialize the output skeletonized image, and construct the
	# morphological structuring element
	area = image.shape[0] * image.shape[1]
	skeleton = np.zeros(image.shape, dtype="uint8")
	elem = cv2.getStructuringElement(structuring, size)

	# keep looping until the erosions remove all pixels from the
	# image
	while True:
		# erode and dilate the image using the structuring element
		eroded = cv2.erode(image, elem)
		temp = cv2.dilate(eroded, elem)

		# subtract the temporary image from the original, eroded
		# image, then take the bitwise 'or' between the skeleton
		# and the temporary image
		temp = cv2.subtract(image, temp)
		skeleton = cv2.bitwise_or(skeleton, temp)
		image = eroded.copy()

		# if there are no more 'white' pixels in the image, then
		# break from the loop
		if area == area - cv2.countNonZero(image):
			break

	# return the skeletonized image
	return skeleton

def opencv2matplotlib(image):
	# OpenCV represents images in BGR order; however, Matplotlib
	# expects the image in RGB order, so simply convert from BGR
	# to RGB and return
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)