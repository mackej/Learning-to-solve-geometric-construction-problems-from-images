from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    O = env.add_free(303.5, 237.5)
    P = env.add_free(385.5, 186)
    env.set_tools("Circle")
    env.goal_params(O, P)

def construct_goals(O, P):
    return circle_tool(O, P)

def get_construction(env, obj):
    O = obj[0]
    P = obj[1]
    return [
        construction.ConstructionProcess('Circle', [O, P]),
    ], [
        circle_tool(O, P)
    ]

def get_tool_hints():
    return [
        "Circle",
    ]
