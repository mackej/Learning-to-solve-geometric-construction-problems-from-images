from py_euclidea.constructions import *
class ConstructionProcess:
    def __init__(self, tool_name, clicks):
        self.tool_name = tool_name
        self.action_clicks = [Point(p.a) for p in clicks]
