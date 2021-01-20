from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction
import numpy as np

def init(env):
    X = env.add_free(318.5, 126.0)
    l = env.add_free_line((5.5, 253.5), (625.0, 253.5))

    env.set_tools(
        "Perpendicular",
    )
    env.goal_params(X, l)

def construct_goals(X, l):
    return perp_tool(l, X)

def additional_bb(X, l, goal):
    return intersection_tool(l, goal)

def get_construction(env, obj):
    X = obj[0]
    l = obj[3]
    g = env.cur_goal()[0]
    p = intersection_tool(g,l)
    p_on_line = Point(p.a + l.v * np.linalg.norm(X.a-p.a)/2)

    return [
        construction.ConstructionProcess('Perpendicular', [p_on_line, X]),
    ], [
        perp_tool(l, X)
    ]
