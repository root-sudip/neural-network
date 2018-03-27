import os

import sys

import glob
import os
import sys
import time

import natsort
import cv2
i = []
path = 'dump4/'
for match in glob.glob("%s/*" % path):
	if match.lower()[-4:] in ('.jpg', '.png', '.gif', 'jpeg'):
		i.append(path+os.path.basename(match))
            #print(os.path.basename(match))

#print(sorted(i))

# k = i.sort(key=lambda f: int(filter(str.isdigit, f)))
# print(k)

k =natsort.natsorted(i)

width = 720
height = 1280
video =  wri = cv2.VideoWriter('vid3.avi',cv2.cv.CV_FOURCC(*'XVID'), 30, (height, width))


for l in k:
	print(l)
	img1 = cv2.imread(l)
	video.write(img1)

cv2.destroyAllWindows()
video.release()
	