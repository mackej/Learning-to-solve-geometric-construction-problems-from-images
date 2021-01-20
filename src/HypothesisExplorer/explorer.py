from tkinter import *
from tkinter.scrolledtext import *
import os
import sys
from HypothesisExplorer.CheckboxGrid import *
from HypothesisExplorer.PictureSelector import *
from LevelSelector import *
import HypothesisExplorer.models_config as config
from MaskRCNNExperiment.InteractiveInferenceModel import *
from py_euclidea import multi_level
from collections import deque
import numpy as np
from PIL import Image
import pickle

class explorer:
    def __init__(self):
        self.choosed_levels= None
        self.choosed_models = None
        self.root = Tk()
        self.top_panel = Frame(self.root, height=100, width=1024, borderwidth=2,)
        #self.top_panel.minsize((1024, 100))
        self.work_panel = Frame(self.root, height=100, width=1224, borderwidth=2, relief=RAISED)
        self.bottom_panel = Frame(self.root, height=100, width=2224, borderwidth=2)
        self.top_panel.pack(side=TOP, fill=BOTH, expand=True)
        self.work_panel.pack(side=TOP)
        self.bottom_panel.pack(side=BOTTOM)
        self.navigate_panel = Frame(self.bottom_panel,height=100, width=628, borderwidth=2)
        self.navigate_panel.pack(side=RIGHT)
        self.console = Frame(self.bottom_panel,height=100, width=628, borderwidth=2)
        self.console.pack(side=LEFT)
        self.root.minsize(1024, 560)
        self.model = None
        self.output_to = sys.stdout
        self.exit_inference = False
        self.console_window = ScrolledText(self.console, height=3, width=135)
        self.console_window.configure(state="disabled")
        self.console_window.pack(side=LEFT)
        class redirect:
            def __init__(self, text):
                self.text = text

            def write(self, s, **kwargs):
                try:
                    self.text.configure(state=NORMAL)
                    self.text.insert("end", s)
                    self.text.configure(state=DISABLED)
                    self.text.update()
                    self.text.see(END)
                except TclError:
                    print(str(s))

            def flush(self):
                pass
        self.redirector = redirect(self.console_window)
        sys.stdout = self.redirector


    def choose_image(self, h_map):
        self.next_lvl = False
        self.reset = False
        l = Label(self.top_panel, text="Click to choose next step of inference.")
        l.config(font=("Courier", 25))
        l.pack(side=TOP)
        grid = None
        k = None

        frame_canvas = Frame(self.work_panel)
        frame_canvas.grid(row=2, column=0, pady=(5, 0), sticky='nw')
        frame_canvas.grid_rowconfigure(0, weight=1)
        frame_canvas.grid_columnconfigure(0, weight=1)
        # Set grid_propagate to False to allow 5-by-5 buttons resizing later
        frame_canvas.grid_propagate(False)

        # Add a canvas in that frame
        canvas = Canvas(frame_canvas)
        canvas.grid(row=0, column=0, sticky="news")

        # Link a scrollbar to the canvas
        vsb = Scrollbar(frame_canvas, orient="vertical", command=canvas.yview)
        vsb.grid(row=0, column=1, sticky='ns')
        canvas.configure(yscrollcommand=vsb.set)

        # Create a frame to contain the buttons
        frame_buttons = Frame(canvas, bg="blue")
        canvas.create_window((0, 0), window=frame_buttons, anchor='nw')

        # Add 9-by-5 buttons to the frame
        if h_map is not None:
            grid = PictureSelector(frame_buttons, h_map, max_col_size=4)
            grid.pack(side=TOP, fill=X)
        else:
            k = Label(frame_buttons, text="There are no hypothesis!")
            k.config(font=("Courier", 25))
            k.pack(side=TOP)
        # Update buttons frames idle tasks to let tkinter calculate buttons sizes
        frame_buttons.update_idletasks()

        first5columns_width = 300 * 4
        first5rows_height = 300 * 1.5
        frame_canvas.config(width=first5columns_width + vsb.winfo_width(),
                            height=first5rows_height)

        # Set the canvas scrolling region
        canvas.config(scrollregion=canvas.bbox("all"))


        def exit_inference():
            self.reset = True
            self.root.quit()
        exit_b = Button(self.navigate_panel, text='exit and change inference setting.', command=exit_inference)
        exit_b.pack(side=BOTTOM)

        def next_level():
            self.next_lvl = True
            self.root.quit()
        next_b = Button(self.navigate_panel, text='Next level', command=next_level)
        next_b.pack(side=BOTTOM)

        self.root.mainloop()
        if self.next_lvl or self.reset:
            frame_canvas.destroy()
            next_b.destroy()
            l.destroy()
            exit_b.destroy()
            return []
        h = grid.get_selected()
        frame_canvas.destroy()
        next_b.destroy()
        l.destroy()
        exit_b.destroy()
        return h

    def choose_levels(self):
        l = Label(self.top_panel, text="Choose levels which should be generated:")
        l.config(font=("Courier", 25))
        l.pack(side=TOP)
        level_packs = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta']
        levels_to_choose = {}

        levels_texts = {}
        for i in level_packs:
            levels_to_choose[i] = LevelSelector.get_levels(match=i + ".*")
            levels_texts[i] = [i[1] for i in levels_to_choose[i]]
        grid = CheckboxGrid(self.work_panel, levels_texts, max_col_size=6)
        grid.pack(side=TOP, fill=X)

        def allstates():
            self.choosed_levels = grid.get_states()
            self.root.quit()

        #Button(root, text='Quit', command=root.quit).pack(side=RIGHT)
        b = Button(self.navigate_panel, text='Choose', command=allstates)
        b.pack(side=RIGHT)
        self.root.mainloop()
        grid.destroy()
        b.destroy()
        l.destroy()
        res = []
        for i in self.choosed_levels.keys():
            for j in range(len(levels_to_choose[i])):
                if self.choosed_levels[i][j]==1:
                    res.append(levels_to_choose[i][j])
        return res

    def choose_models(self):
        l = Label(self.top_panel, text="Choose models which should be loaded for inference.")
        l.config(font=("Courier", 25))
        l.pack(side=TOP)
        names_of_models = {}
        for i in config.avaliable_models.keys():
            names_of_models[i] = [key for key in config.avaliable_models[i].keys()]

        grid = CheckboxGrid(self.work_panel, config.avaliable_models, max_col_size=6)
        grid.pack(side=TOP, fill=X)

        def allstates():
            self.choosed_models = grid.get_states()
            self.root.quit()

        # Button(root, text='Quit', command=root.quit).pack(side=RIGHT)
        b = Button(self.navigate_panel, text='Choose', command=allstates)
        b.pack(side=RIGHT)
        self.root.mainloop()
        grid.destroy()
        b.destroy()
        l.destroy()
        res = []
        names = []
        for i in self.choosed_models.keys():
            for j in range(len(config.avaliable_models[i])):
                if self.choosed_models[i][j] == 1:
                    res.append(list(config.avaliable_models[i].values())[j])
                    names.append([i, list(config.avaliable_models[i].keys())[j]])
        return res, names
    def enable_disable_models(self):
        l = Label(self.top_panel, text="Enable or disable models for detection")
        l.config(font=("Courier", 25))
        l.pack(side=TOP)
        names_of_models = {}
        for i in config.avaliable_models.keys():
            names_of_models[i] = [key for key in config.avaliable_models[i].keys()]

        grid = CheckboxGrid(self.work_panel, config.avaliable_models, max_col_size=6)
        initialy_enabled = []
        for i in range(len(self.model.enabled)):
            if self.model.enabled[i]:
                initialy_enabled.append(self.model_names[i])

        grid.only_one_enabled(self.model_names)
        grid.pre_selected(initialy_enabled)
        grid.pack(side=TOP, fill=X)

        def allstates():
            self.choosed_models = grid.get_states()
            self.root.quit()

        # Button(root, text='Quit', command=root.quit).pack(side=RIGHT)
        b = Button(self.navigate_panel, text='Choose', command=allstates)
        b.pack(side=RIGHT)
        self.root.mainloop()
        grid.destroy()
        b.destroy()
        l.destroy()
        self.model.enabled = grid.gather(self.model_names)
    def load_weigths(self, models, model_names):
        self.model = InteractiveInferenceMultiModel.load_model_exact_path([os.path.join("logs",i) for i in models], output_to=self.redirector)
        self.model_names = model_names
    def run_inference(self, levels):
        self.exit_inference = False
        m = multi_level.MultiLevel((
            levels
        ))
        history_size = 1
        additional_moves = 2
        history = deque(maxlen=history_size)

        source_name_map = {}
        for i in range(len(self.model_names)):
            source_name_map[self.model.model[i].model_path] = self.model_names[i]

        while True:
            done = False
            level_index = m.next_level()
            for i in range(history_size):
                history.append(np.zeros(m.out_size))
            while done is not True:
                image = env_utils.EnvironmentUtils.build_image_from_multilevel(m, history)
                pred = self.model.detect([image], verbose=0, bool_masks=False, high_level_verbose=1)

                last_state = image[:, :, 0]
                history.append(last_state)
                hypothesis_map = self.model.build_hypothesis_visualisations(pred, None, m)
                i = 0
                for h in hypothesis_map:
                    h["hypothesis"]["model_name"] = source_name_map[h["hypothesis"]["source"]]
                    i+=1
                    #img = Image.fromarray(h["image"], 'RGB')
                    #img.save('{}_{}_{}.png'.format(h["hypothesis"]["model_name"][1],h["hypothesis"]["score"],i))
                choosed_h = self.choose_image(hypothesis_map)
                if self.next_lvl:
                    break
                if self.reset:
                    return
                choosed_h = choosed_h["hypothesis"]
                for s in choosed_h["steps"]:
                    m.cur_env.add_and_run(s)
                r, done = m.evaluate_last_step(size=len(choosed_h["steps"]))

                if r is None and done is None:
                    break


