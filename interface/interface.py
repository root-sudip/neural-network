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

def images():
    im = []
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            im.extend(images_for(path))
    else:
        im.extend(images_for(os.getcwd()))
    return natsort.natsorted(im)
    #return sorted(im, key=lambda s:s.lower())

def images_for(path):
    if os.path.isfile(path):
        return [path]
    i = []
    for match in glob.glob("%s/*" % path):
        if match.lower()[-4:] in ('.jpg', '.png', '.gif'):
            
            i.append(match)
    return i

class ExampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0

        #as_array = np.asarray(Image.open('my_drawing.jpg').resize((512,512), Image.ANTIALIAS))
        #shape = as_array.shape
        self._images = images()
        self.position = 0

        self.canvas = tk.Canvas(self, width=1280, height=720, cursor="cross")
        self.canvas.pack(side="top", fill="both", expand=True)

        self.im = Image.open(self._images[self.position]).resize((1280,720), Image.ANTIALIAS)
        self.canvas.image = ImageTk.PhotoImage(self.im)
        self.canvas.create_image(0, 0, image=self.canvas.image,anchor='nw')
        
        # button = tk.Button(text='Next')

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind_all("<KeyPress-Right>", self.show_next_image)
        self.canvas.bind_all("<KeyPress-Left>", self.save_image)
        self.rect = None


    def show_next_image(self,event):
        if event.keysym == 'Right':
            print(self._images[self.position])
            self.position = self.position + 1
            self.im = Image.open(self._images[self.position]).resize((1280,720), Image.ANTIALIAS)
            self.canvas.image = ImageTk.PhotoImage(self.im)
            self.canvas.create_image(0, 0, image=self.canvas.image,anchor='nw')
        elif event.keysym == 's':
            print('s')
            with open("a.csv","a") as file:
                file.write(self.x0,",",self.y0,",",self.x1,",",self.y1)
                file.write("\n")
                file.close()
    def save_image(self,event):
        if event.keysym == 'Left':
            print('Left')
            with open("a.csv","a") as file:
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

        self.canvas.create_rectangle(self.x0,self.y0,self.x1,self.y1, outline="black",width=3)
        #filename = "my_drawing.jpg"
        #self.im.save(filename)
        print('c')
        print(self.x0,self.y0,self.x1,self.y1)


    # def on_move_press(self, event):
    #     curX = self.canvas.canvasx(event.x)
    #     curY = self.canvas.canvasy(event.y)

    #     w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
    #     if event.x > 0.9*w:
    #         self.canvas.xview_scroll(1, 'units') 
    #     elif event.x < 0.1*w:
    #         self.canvas.xview_scroll(-1, 'units')
    #     if event.y > 0.9*h:
    #         self.canvas.yview_scroll(1, 'units') 
    #     elif event.y < 0.1*h:
    #         self.canvas.yview_scroll(-1, 'units')

    #     # expand rectangle as you drag the mouse
    #     self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)

if __name__ == "__main__":
    app = ExampleApp()
    app.mainloop()



# from tkinter import Tk, Canvas
# from PIL import ImageTk, Image
# root = Tk()
# #Create a canvas
# canvas = Canvas(root, width=400, height=300)
# canvas.pack()
# im = Image.open('08fec7a.jpg')
# canvas.image = ImageTk.PhotoImage(im)
# canvas.create_image(0, 0, image=canvas.image, anchor='nw')
# root.mainloop()