"""
Developer: Sudip Das
Licence : Indian Statistical Institute
"""


import tkinter as tk
from PIL import ImageTk, Image
import os
import numpy as np

class ExampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0

        as_array = np.asarray(Image.open('my_drawing.jpg'))
        shape = as_array.shape


        self.canvas = tk.Canvas(self, width=shape[1], height=shape[0], cursor="cross")
        self.canvas.pack(side="top", fill="both", expand=True)

        self.im = Image.open('my_drawing.jpg')
        self.canvas.image = ImageTk.PhotoImage(self.im)
        self.canvas.create_image(0, 0, image=self.canvas.image, anchor='nw')

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        
    def on_button_press(self, event):
        self.x = event.x
        self.y = event.y

    def on_button_release(self, event):
        x0,y0 = (self.x, self.y)
        x1,y1 = (event.x, event.y)

        self.canvas.create_rectangle(x0,y0,x1,y1, outline="black",width=3)
        #filename = "my_drawing.jpg"
        #self.im.save(filename)

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