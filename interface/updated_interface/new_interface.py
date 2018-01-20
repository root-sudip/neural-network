"""
Developer: Sudip Das
Licence : Indian Statistical Institute
sudo apt-get install python-imaging-tk #for pyhton2
python3-pil.imagetk # fo python3

 To run this, pyhton3 interface.py inputdir/ outputdir/ csvfilename
"""


import tkinter as tk
from PIL import ImageTk, Image
import os
import numpy as np
import sys

import glob
import os
import sys
import time

import natsort


height = 623
width = 623


#csv_filename = 

def images():
    im = []
    if len(sys.argv) > 1:
        for path in sys.argv[1:2]:
            im.extend(images_for(path))
    else:
        im.extend(images_for(os.getcwd()))
    return natsort.natsorted(im)


def images_for(path):
    if os.path.isfile(path):
        return [path]
    i = []
    for match in glob.glob("%s/*" % path):
        if match.lower()[-4:] in ('.jpg', '.png', '.gif', 'jpeg'):
            i.append(path+os.path.basename(match))
            #print(os.path.basename(match))
    return i



def images_K():
    im = []
    if len(sys.argv) > 1:
        for path in sys.argv[2:]:
            im.extend(images_for(path))
    else:
        im.extend(images_for_k(os.getcwd()))
    return natsort.natsorted(im)


def images_for_K(path):
    if os.path.isfile(path):
        return [path]
    i = []
    for match in glob.glob("%s/*" % path):
        if match.lower()[-4:] in ('.jpg', '.png', '.gif', 'jpeg'):
            i.append(path+os.path.basename(match))
            #print(os.path.basename(match))
    return i





class ExampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        self._images = images()
        self.total_imges = len(self._images) - 1
        
        self.position = 0

####start
        self.imges = images_K()

        if self.imges == []:
           self.imgk = 0 
        else:
 
           temp = self.imges[-1]
           #print('==>',os.path.basename(temp).strip(".png"))
           self.imgk = int(os.path.basename(temp).strip(".png")) + 1
	
###end
        self.canvas = tk.Canvas(self, width=623, height=623, cursor="cross")
        self.canvas.pack(side="top", fill="both", expand=True)


        self.im = Image.open(self._images[self.position]).resize((height,width), Image.ANTIALIAS)
        self.canvas.image = ImageTk.PhotoImage(self.im)
        self.canvas.create_image(0, 0, image=self.canvas.image,anchor='nw')


        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind_all("<KeyPress-Right>", self.show_next_image)
        self.canvas.bind_all("<KeyPress-Left>", self.show_prev_image)
        self.canvas.bind_all("<KeyPress-s>", self.save_image)
        self.rect = None


    def show_next_image(self,event):
        if event.keysym == 'Right':
            if self.position < self.total_imges:
                print(self._images[self.position])
                self.position = self.position + 1
                print("self.position",self.position)
                self.im = Image.open(self._images[self.position]).resize((623,623), Image.ANTIALIAS)
                self.canvas.image = ImageTk.PhotoImage(self.im)
                self.canvas.create_image(0, 0, image=self.canvas.image,anchor='nw')
        else:
            pass
            

    def show_prev_image(self,event):
        if event.keysym == 'Left' and self.position >= 0 :
            print(self._images[self.position])
            self.position = self.position - 1
            self.im = Image.open(self._images[self.position]).resize((623,623), Image.ANTIALIAS)
            self.canvas.image = ImageTk.PhotoImage(self.im)
            self.canvas.create_image(0, 0, image=self.canvas.image,anchor='nw')
        else:
            pass
            

    def save_image(self,event):
        
        if event.keysym == 's':
            print('saving coordinate with image name',self._images[self.position])
            img = Image.open(self._images[self.position])
            crop = img.crop((self.x0,self.y0,self.x1,self.y1))
            crop.save('out/'+str(self.imgk)+'.png')
            self.imgk = self.imgk + 1
            with open("a.csv","a") as file:

                file.write(self._images[self.position])
                file.write(",")
                file.write(str(self.x0))
                file.write(",")
                file.write(str(self.y0))
                file.write(",")
                file.write(str(self.x1))
                file.write(",")
                file.write(str(self.y1))
                file.write("\n")
                file.close()


    def on_button_press(self, event):
        self.x = event.x
        self.y = event.y

    def on_button_release(self, event):
        self.x0,self.y0 = (self.x, self.y)
        self.x1,self.y1 = (event.x, event.y)
        self.canvas.create_rectangle(self.x0,self.y0,self.x1,self.y1, outline="red",width=2)
        print(self.x0,self.y0,self.x1,self.y1)



if __name__ == "__main__":
    app = ExampleApp()
    app.mainloop()


