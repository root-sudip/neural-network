"""
Developer: Sudip Das
Licence : Indian Statistical Institute
"""

# import os
# import re
# import json
# import glob
# import cv2 as cv
# from collections import defaultdict

# from PIL import Image

# import sys

# annotations = json.load(open('annotations.json'))

# out_dir = 'data/plots'
# if not os.path.exists(out_dir):
#     os.makedirs(out_dir)

# img_fns = defaultdict(dict)

# for fn in sorted(glob.glob('images/*.png')):
#     set_name = re.search('(set[0-9]+)', fn).groups()[0]
#     img_fns[set_name] = defaultdict(dict)

# for fn in sorted(glob.glob('images/*.png')):
#     set_name = re.search('(set[0-9]+)', fn).groups()[0]
#     video_name = re.search('(V[0-9]+)', fn).groups()[0]
#     img_fns[set_name][video_name] = []

# for fn in sorted(glob.glob('images/*.png')):
#     set_name = re.search('(set[0-9]+)', fn).groups()[0]
#     video_name = re.search('(V[0-9]+)', fn).groups()[0]
#     n_frame = re.search('_([0-9]+)\.png', fn).groups()[0]
#     img_fns[set_name][video_name].append((int(n_frame), fn))

# n_objects = 0

# k = 0


# height = 640
# width = 480

# for set_name in sorted(img_fns.keys()):
#     for video_name in sorted(img_fns[set_name].keys()):
#         wri = cv.VideoWriter(
#             'data/plots/{}_{}.avi'.format(set_name, video_name),
#             cv.cv.CV_FOURCC(*'XVID'), 30, (640, 480))
#         for frame_i, fn in sorted(img_fns[set_name][video_name]):
#             img = cv.imread(fn)
#             if str(frame_i) in annotations[set_name][video_name]['frames']:
#                 data = annotations[set_name][
#                     video_name]['frames'][str(frame_i)]
#                 for datum in data:
#                     x, y, w, h = [int(v) for v in datum['pos']]
#                     cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
#                     ###
#                     #I have img and two coordinates also
#                     for i in range(0,height,200):
#                         for j in range(0,width,200):
#                             try:
#                                 if (i > x and j > y) and (i + 200 > x + w and j + 200 > y + h):
#                                     crop = img[i:i+200,j:j+200]
#                                     im = Image.fromarray(crop)
#                                     im.save(sys.argv[1]+"image_"+str(k)+".png")
#                                     k = k + 1
#                             except ValueError:
#                                 pass
#                     ###
#                     n_objects += 1
#                 wri.write(img)
#             else:
#                 for i in range(0,height,200):
#                         for j in range(0,width,200):
#                             try:
#                                 crop = img[i:i+200,j:j+200]
#                                 im = Image.fromarray(crop)
#                                 im.save(sys.argv[1]+"image_"+str(k)+".png")
#                                 k = k + 1
#                             except ValueError:
#                                 pass
#         wri.release()
#         print(set_name, video_name)
# print(n_objects)



import os
import re
import json
import glob
import cv2 as cv
from collections import defaultdict

from PIL import Image

import sys

annotations = json.load(open('annotations.json'))

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
        wri = cv.VideoWriter(
            'data/plots/{}_{}.avi'.format(set_name, video_name),
            cv.cv.CV_FOURCC(*'XVID'), 30, (640, 480))
        for frame_i, fn in sorted(img_fns[set_name][video_name]):
            img = cv.imread(fn)
            if str(frame_i) in annotations[set_name][video_name]['frames']:
                data = annotations[set_name][
                    video_name]['frames'][str(frame_i)]
                for datum in data:
                    x, y, w, h = [int(v) for v in datum['pos']]
                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    n_objects += 1
                    # x1 = range(x,x+h)
                    # y1 = range(y,y+w)
                wri.write(img)
                    ###
                    #I have img and two coordinates also
            else:
                for i in range(0,height,100):
                    for j in range(0,width,100):
                        try:
                            #x1 = range(i,i+200)
                            #y1 = range(j,j+200)

                            #x2 = set(range(x,x+w))
                            #y2 = set(range(y,y+h))
                                #if (i > x or j > y) or (i + 200 > x + w or j + 200 > y + h):
                            #if (len(x2.intersection(x1)) == 0) and (len(y2.intersection(y1)) == 0):
                                #print(len(x2.intersection(x1)))
                                #print(len(y2.intersection(y1)))
                            crop = img[i:i+100,j:j+100]
                            im = Image.fromarray(crop)
                            im.save(sys.argv[1]+"image_"+str(k)+".png")
                            k = k + 1
                        except ValueError:
                            pass
                    ###
                    
                
            # else:
            #     for i in range(0,height,200):
            #             for j in range(0,width,200):
            #                 try:
            #                     crop = img[i:i+200,j:j+200]
            #                     im = Image.fromarray(crop)
            #                     im.save(sys.argv[1]+"image_"+str(k)+".png")
            #                     k = k + 1
            #                 except ValueError:
            #                     pass
        wri.release()
        print(set_name, video_name)
print(n_objects)