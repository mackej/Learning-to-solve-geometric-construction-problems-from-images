class Action:
    def __init__(self, tool, tool_name):
        self.tool = tool
        self.tool_name = tool_name
        self.in_objects = []
        #how many object will action create during its construction... used for reversing action
        self.action_obj_cnt = 1
        self.action_reward = None
        self.additional_point_actions = []

    def __str__(self):
        return "<tool='{}' on objects: {} reward={}".format(self.tool_name, str(self.in_objects), self.action_reward)
