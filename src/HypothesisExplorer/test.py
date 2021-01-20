from tkinter import  *
import numpy as np
from PIL import Image, ImageTk
import pickle

root = Tk()
h_map = pickle.load(open("save.p", "rb"))
l = Label(root, text="Choose next step of construction from hypothesis")
l.pack(side=TOP)
for i in h_map:
    img = Image.fromarray(i["image"])
    img = img.resize((500, 500))
    tk_img = ImageTk.PhotoImage(image=img)
    my_img = Button(root, image=tk_img)
    my_img.pack(side=TOP)
root.mainloop()