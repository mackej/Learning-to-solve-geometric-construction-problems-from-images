from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction
import numpy as np

def init(env):
    C = env.add_free(350.0, 235.0)
    l = env.add_free_line((1.0, 340.0), (634.5, 331.0))

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "intersection", "Perpendicular",
    )
    env.goal_params(C, l)

def construct_goals(C, l):
    r = l.dist_from(C.a)
    return Circle(C.a, r)

def get_construction(env, obj):
    C = obj[0]
    l = obj[3]
    X = point_on(C.a, l)
    T = intersection_tool(l, perp_tool(l, C))
    return [
        construction.ConstructionProcess('Perpendicular', [X, C]),
        construction.ConstructionProcess('Circle', [C, T]),
        construction.ConstructionProcess('Circle', [C, T]),
    ], [
        perp_tool(l, C),
        T,
        circle_tool(C, T)
    ]
