from py_euclidea.geo_object import Point
import cairo
import numpy as np
import random as rnd

from py_euclidea.ConstructionProcess import  ConstructionProcess
from py_euclidea.geo_object import corners_to_rectangle
from py_euclidea.environment import load_level
from py_euclidea.tools import key_to_tool

class MultiLevel:
    def __init__(
        self, levels,
        drop_prob = 0, max_moves = 20,
        out_size = (256,256), scale = (2,2),
    ):
        self.out_size = np.array(out_size, dtype = int)
        self.scale = np.array(scale, dtype = float)
        self.corners = np.array((
            (0, 0),
            out_size,
        )) * self.scale
        self.levels = [
            load_level(level_pack, level, self.out_size * self.scale)
            for (level_pack, level) in levels
        ]
        self.drop_prob = drop_prob
        self.max_moves = max_moves
        self.cur_env = None

        self.tools = []
        self.tool_name_to_index = dict()
        self.tool_index_to_name = dict()
        for tool, name in key_to_tool.values():
            if name == "move": continue
            index = len(self.tools)
            self.tools.append(tool)
            self.tool_name_to_index[name] = index
            self.tool_index_to_name[index] = name
        # self.start_level()

    def get_min_set_of_tool(self):
        available_tools = {}
        for l in self.levels:
            for tool_name in l.enabled_tools:
                if tool_name in ["move", "intersection"]:
                    continue
                available_tools[tool_name] = True
        return available_tools.keys()

    def get_min_set_of_tools_for_constructions(self):
        available_tools = {}
        for l in range(len(self.levels)):
            self.start_level(level_index=l)
            for c in self.cur_env.construction:
                available_tools[c.tool_name] = True
        return available_tools

    def start_level(self, level_index=None, logged_level=None):
        if level_index is None:
            level_index = rnd.randrange(len(self.levels))
        #self.cur_env = rnd.choice(self.levels)

        if logged_level is not None:
            self.cur_env = self.levels[logged_level.level_index]

            for i in range(len(self.cur_env.objs)):
                if hasattr(self.cur_env.objs[i], "index"):
                    logged_level.objs[i].set_index(self.cur_env.objs[i].index)
                if hasattr(self.cur_env.objs[i], "hidden"):
                    logged_level.objs[i].set_hidden(self.cur_env.objs[i].hidden)
            self.cur_env.objs = logged_level.objs
            # When we loading levels from unfinished levels, we only consider single goal.
            # Goal chosen while creating this task, that is because some level goal may collide each other.
            self.cur_env.goals = [logged_level.goal]
            self.cur_env.goal_index = 0
            self.cur_env.construction, self.cur_env.construction_objects = self.cur_env.generate_construction(self.cur_env, self.cur_env.objs)
            self.remaining_goals = logged_level.goal
        else :

            self.cur_env = self.levels[level_index]
            self.cur_env.rnd_init()
            self.remaining_goals = list(self.cur_env.cur_goal())

        self.tool = None
        self.moves = 0

        self.goal_reward = 1. / len(self.remaining_goals)

        self.tool_mask = np.zeros(len(self.tools), dtype = bool)
        for tool_name in self.cur_env.enabled_tools:
            if tool_name == "move": continue
            tool_index = self.tool_name_to_index[tool_name]
            self.tool_mask[tool_index] = True
        return level_index

    def stop_level(self):
        if self.cur_env is not None:
            self.cur_env.restart()
        self.cur_env = None

    def next_level(self, level_index=None, logged_level=None):
        self.stop_level()
        return self.start_level(level_index=level_index, logged_level=logged_level)

    def objects_to_numpy(self, objs):
        width, height = self.out_size
        surface = cairo.ImageSurface(cairo.FORMAT_A8, width, height)
        cr = cairo.Context(surface)
        cr.scale(*(1/self.scale))

        #cr.rectangle(*corners_to_rectangle(self.corners))
        #cr.set_source_rgb(0, 0, 0)
        #cr.fill()

        #cr.arc(5, 5, 4, 0, 2*np.pi)
        #cr.set_line_width(2)
        #cr.set_source_rgb(1,1,1)
        #cr.stroke()

        cr.set_source_rgb(1, 1, 1)
        #Point([10, 10]).draw(cr, self.corners, 1)
        for obj in objs:
            obj.draw(cr, self.corners, 1)

        data = surface.get_data()
        data = np.array(data, dtype = np.uint8)
        data = data.reshape([height, surface.get_stride()])
        data = data[:,:width]
        return data.T

    def get_construction_length(self):
        return len(self.cur_env.construction)

    def get_construction(self, step_index):
        if step_index >= len(self.cur_env.construction) or step_index < 0:
            return None, None
        step = self.cur_env.construction[step_index]
        pts = [i.a/self.scale for i in step.action_clicks]
        tool = self.tool_name_to_index[step.tool_name]
        return tool, pts

    def get_state(self):

        ori_layer = self.objects_to_numpy(self.cur_env.objs[:self.cur_env.min_objs])
        visible = []
        for i in self.cur_env.objs:
            if not hasattr(i, 'hidden') or i.hidden is False:
                visible.append(i)
        all_layer = self.objects_to_numpy(visible)
        goal_layer = self.objects_to_numpy(self.cur_env.cur_goal())
        if self.tool is None: selected_objs = ()
        else: selected_objs = self.tools[self.tool].get_highlighted(self.cur_env)
        sel_layer = self.objects_to_numpy(selected_objs)

        return np.stack((ori_layer, all_layer, goal_layer, sel_layer), axis = -1)

    def action_set_tool(self, tool_index):
        #assert(tool is None)
        assert(tool_index >= 0
               and tool_index < len(self.tools)
               and self.tool_mask[tool_index])
        self.tool = tool_index
        self.tools[tool_index].initialize(self.cur_env)

    def action_click_point(self, coor, auto_proceed=False, change_rem_goals=True):
        if (self.corners[0] / self.scale > coor).any() or (coor > self.corners[1] / self.scale).any():
            raise Exception("action is outside bounding box!")
        return self.action_click(coor[0], coor[1], auto_proceed, change_rem_goals=change_rem_goals)

    def action_click(self, x, y, auto_proceed=False, change_rem_goals=True):
        '''
        :param x: x coor of click
        :param y: y coor of click
        :param auto_proceed: False do not proceed, True load ext level when done
        :param change_rem_goals: change rem goal while clicking. Use False only when we plan to revert this action
        :return:
        '''
        coor = np.array([x, y])*self.scale
        tool_status = self.tools[self.tool].run(self.cur_env, coor, 1, (self.scale[0]) * 10)
        if tool_status is False: return 0, False, tool_status

        finish = False
        reward = 0
        if tool_status is False: reward -= 0.1
        else:
            reward, finish = self.evaluate_last_step(change_rem_goals=change_rem_goals)
            
        self.moves += 1
        if self.moves == self.max_moves: finish = True
        elif self.drop_prob > 0 and self.drop_prob > np.random.random(): finish = True

        if finish and auto_proceed:
            print("auto_proceeded")
            self.next_level()

        return reward, finish, tool_status

    def evaluate_last_step(self, change_rem_goals=True, size=None):
        reward = 0
        finish = False
        num_outs = len(self.cur_env.steps[-1].otypes)
        if size is not None:
            num_outs = size
        tool_output = self.cur_env.objs[-num_outs:]
        rem_goals_next = []
        for goal in self.remaining_goals:
            if any(goal.identical_to(out) for out in tool_output):
                reward += self.goal_reward
            else:
                rem_goals_next.append(goal)
        if change_rem_goals:
            self.remaining_goals = rem_goals_next
        if len(self.remaining_goals) == 0:
            finish = True
        return reward, finish
