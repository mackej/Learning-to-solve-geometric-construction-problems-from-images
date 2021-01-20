#!/usr/bin/python3
from tkinter import *
import numpy as np
from PIL import Image, ImageTk

class Picture_Box(Frame):
    def __init__(self, parent=None,index=0, data=None, size=(300,300)):
        self.index = index
        self.parent = parent
        Frame.__init__(self, parent)
        img = Image.fromarray(data["image"])
        img = img.resize(size)
        self.tk_img = ImageTk.PhotoImage(image=img)
        l = Label(self, text="{} (score:{:.2f}, reward:{:.2f})".format(data["tool_name"], data["hypothesis"]["score"], data["hypothesis"]["reward"]))
        l.pack(side=TOP)
        l2 = Label(self, text="source: {}".format(data["hypothesis"]["model_name"]))
        l2.config(font=("Arial", 6))
        l2.pack(side=TOP)
        b = Button(self, image=self.tk_img, text="click me", command=self.pick)
        b.pack(side=TOP)
    def pick(self):
        self.parent.pick(self.index)

class PictureSelector(Frame):
   def __init__(self, parent=None, h_map=[], max_col_size=3):
      Frame.__init__(self, parent)

      self.choosed_h = None
      self.h_map = h_map
      l = Label(self, text="Choose next step of construction from hypothesis")
      l.grid(row=0, column=0)
      self.images = []
      for i in range(len(h_map)):
        p = Picture_Box(self, index=i, data=h_map[i])
        p.grid(row=int(i/max_col_size), column=int(i%max_col_size))
   def pick(self, index):
       self.choosed_h = self.h_map[index]
       self.master.quit()

   def get_selected(self):
       return self.choosed_h



