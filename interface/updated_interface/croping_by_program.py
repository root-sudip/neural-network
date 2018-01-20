"""python3 croping_by_program.py a.csv out1/"""


import csv
import sys
from PIL import Image

file_name = sys.argv[1]

out_name = sys.argv[2]

with open(file_name) as fl:
        line = csv.reader(fl)
        k = 0
        for row in line:
            im = Image.open(row[0])
            im1 = im.crop((int(row[1]),int(row[2]),int(row[3]),int(row[4])))
            im1.save(out_name+str(k)+".png")
            k = k + 1