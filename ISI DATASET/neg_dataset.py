
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

out_dir = 'data/plots'
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
        kl = 0
        for frame_i, fn in sorted(img_fns[set_name][video_name]):
            img = cv.imread(fn)

            if kl <= 1200:


                #added
                im = Image.fromarray(img)

                if str(frame_i) in annotations[set_name][video_name]['frames']:
                    data = annotations[set_name][video_name]['frames'][str(frame_i)]
                    print(fn)


                    try:
                        for i in range(0,height,150):
                            for j in range(0,width,150):
                                rect1=[i,j,i+150,j+150]

                                for datum in data:
                                    x, y, w, h = [int(v) for v in datum['pos']]

                                    rect2=[x,y,x+w,y+h]

                                    if find_overlap(rect1,rect2):
                                        f = 0
                                        break
                                    else:
                                        f = 1

                                if f == 1 and i+150 <height and j+150 < width:
                                
                                    z = im.crop((i, j, i+150, j+150))
                                    z.save('training_v2/negative/'+str(k)+'.png')
                                    k = k + 1   
            
                    except ValueError:
                        pass
            else:
                break

            kl = kl + 1
                

 
print('Total negative samples : ',k)