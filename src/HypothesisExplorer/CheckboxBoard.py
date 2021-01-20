#!/usr/bin/python3
from tkinter import *
class CheckboxBoard():
   def __init__(self, parent=None, picks=[], side=LEFT, anchor=W, row=1, board_name='all', max_col_size=7):
      self.board_name = board_name
      self.all_chk_var = IntVar()
      self.all_chk = Checkbutton(parent, text=board_name, variable=self.all_chk_var, command=self.check_all, anchor=anchor)
      self.all_chk.grid(row=row, column=0, sticky="W")
      self.all_chk.config(relief=GROOVE, bd=2)
      #self.all_chk.pack(side=side, anchor=anchor, expand=YES)
      self.vars = {}
      self.ch_boxes = {}
      c = 0
      self.span_size = 1
      for pick in picks:
         var = IntVar()
         ch = Checkbutton(parent, text=pick, variable=var, anchor=anchor)
         ch.grid(row=row, column=c+1, sticky="W")
         #ch.pack(side=side, anchor=anchor, expand=YES)
         self.vars[pick] = var
         self.ch_boxes[pick]=ch
         c+=1
         if max_col_size <= c:
             c = 0
             row += 1
             self.span_size +=1

      #self.config(relief=GROOVE, bd=2)
   def state(self):
      return map((lambda var: var.get()), self.vars.values())
   def check_all(self):
       for ch in self.ch_boxes.values():
           if ch.cget('state') == 'disabled':
               continue
           if self.all_chk_var.get() == 1:
               ch.select()
           else:
               ch.deselect()