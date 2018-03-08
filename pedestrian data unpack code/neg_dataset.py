
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