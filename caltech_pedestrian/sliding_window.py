from pyimagesearch.helpers import pyramid
from pyimagesearch.helpers import sliding_window
import argparse
import time
import cv2

from PIL import Image

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image and define the window width and height
image = cv2.imread(args["image"])
(winW, winH) = (128, 128)

global clone
global resized

# loop over the image pyramid
try:
	for resized in pyramid(image, scale=0):
		# loop over the sliding window for each layer of the pyramid
		for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
			# if the window does not meet our desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue

			cv2.rectangle(resized, (x, y), (x + winW, y + winH), (0, 255, 0), 2)

except ZeroDivisionError:
	pass

# print(clone)
img_s = Image.fromarray(resized)
img_s.save("sliding_img.png")

