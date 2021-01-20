#!/usr/bin/python3
from tkinter import *
from HypothesisExplorer.CheckboxBoard import *


class CheckboxGrid(Frame):
    def __init__(self, parent=None, boards={}, max_col_size=7):
        Frame.__init__(self, parent)

        row = 0
        self.sub_boards = {}
        for b in boards.keys():
            board = CheckboxBoard(self, boards[b], row=row, board_name=b, max_col_size=max_col_size)
            # board.pack(side=TOP, fill=X)
            self.sub_boards[b] = board
            row += board.span_size + 1
            self.grid_rowconfigure(row - 1, minsize=15)

    def only_one_enabled(self, enabled):
        for i in self.sub_boards.values():
            for j in i.ch_boxes.values():
                j.config(state=DISABLED)
        for i,j in enabled:
            self.sub_boards[i].ch_boxes[j].config(state=NORMAL)

    def pre_selected(self, selected):
        for i,j in selected:
            self.sub_boards[i].ch_boxes[j].select()

    def gather(self, to_gather):
        res = []
        for i, j in to_gather:
            res.append(self.sub_boards[i].vars[j].get()>0)
        return res
    def get_states(self):
        res = {}
        for i in self.sub_boards.values():
            res[i.board_name] = list(i.state())
        return res
