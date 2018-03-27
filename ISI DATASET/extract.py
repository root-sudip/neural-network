import cv2
vidcap = cv2.VideoCapture('video_05.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  cv2.imwrite("video_05/set_01_video_05_frame%d.png" % count, image)     # save frame as JPEG file
  count += 1
print 'Total number : ',count