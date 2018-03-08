
"""
Developer: Sudip Das
Licence : Indian Statistical Institute
"""

import os
import re
import json
import glob
import cv2 as cv
from collections import defaultdict

from PIL import Image

import sys

from FindOverlap import *

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

#iou =bb_intersection_over_union([10,30,30,40],[10,40,30,60])
#print(iou)




annotations = json.load(open('annotations/annotations.json'))

out_dir = 'dataa/plots'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

img_fns = defaultdict(dict)

for fn in sorted(glob.glob('data/images/*.png')):
    set_name = re.search('(set[0-9]+)', fn).groups()[0]
    img_fns[set_name] = defaultdict(dict)

for fn in sorted(glob.glob('data/images/*.png')):
    set_name = re.search('(set[0-9]+)', fn).groups()[0]
    video_name = re.search('(V[0-9]+)', fn).groups()[0]
    img_fns[set_name][video_name] = []

for fn in sorted(glob.glob('data/images/*.png')):
    set_name = re.search('(set[0-9]+)', fn).groups()[0]
    video_name = re.search('(V[0-9]+)', fn).groups()[0]
    n_frame = re.search('_([0-9]+)\.png', fn).groups()[0]
    img_fns[set_name][video_name].append((int(n_frame), fn))

n_objects = 0

k = 0


height = 640
width = 480

for set_name in sorted(img_fns.keys()):
    for video_name in sorted(img_fns[set_name].keys()):
        for frame_i, fn in sorted(img_fns[set_name][video_name]):
            img = cv.imread(fn)

            #added
            im = Image.fromarray(img)

            if str(frame_i) in annotations[set_name][video_name]['frames']:
                data = annotations[set_name][video_name]['frames'][str(frame_i)]
                print(fn)


                try:
                    for i in range(0,height,64):
                        for j in range(0,width,64):
                            rect1=[i,j,i+64,j+64]

                            for datum in data:
                                x, y, w, h = [int(v) for v in datum['pos']]

                                rect2=[x,y,x+w,y+h]

                                if find_overlap(rect1,rect2):
                                    f = 0
                                    break
                                else:
                                    f = 1

                            if f == 1:
                                z = im.crop((i, j, i+64, j+64))
                                z.save('dumped/'+str(k)+'.png')
                                k = k + 1                   
                except ValueError:
                    pass
print('total number of negative samples : ',k)