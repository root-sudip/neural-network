"""
Developer: Sudip Das
Licence : Indian Statistical Institute
"""

# import cv2
# print(cv2.__version__)
# vidcap = cv2.VideoCapture('DrivingDowntown-FifthAvenue-NewYorkCityNYUSA.mp4')
# success,image = vidcap.read()
# count = 0
# success = True
# while success:
#   success,image = vidcap.read()
#   print 'Read a new frame: ', success
#   cv2.imwrite("data/%d.jpg" % count, image)     # save frame as JPEG file
#   count += 1
import cv2
import math

videoFile = "DrivingDowntown-FifthAvenue-NewYorkCityNYUSA.mp4"
imagesFolder = "data/"
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5) #frame rate
count = 0
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename = imagesFolder  +  str(int(frameId)) + ".jpg"
        cv2.imwrite(filename, frame)
        count = count + 1
        print "%d\r"%count
cap.release()
print "Done!"