from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction
import numpy as np

def init(env):
    X = env.add_free(339.0, 276.0)
    Y = env.add_free(2.0, 277.5, hidden = True)
    l = env.add_line(X, Y)

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "intersection",
    )
    env.goal_params(X, l)

def construct_goals(X, l):
    return perp_tool(l, X)

def get_construction(env, obj):
    X = obj[0]
    l = obj[2]
    A = Point(X.a + l.v * 100)
    B = Point(X.a - l.v * 100)
    return [
        construction.ConstructionProcess('Circle', [X, A]),
        construction.ConstructionProcess('Circle', [X, A]),
        construction.ConstructionProcess('Perpendicular_Bisector', [A,B]),
        construction.ConstructionProcess('Perpendicular_Bisector', [A,B]),
    ], [
        A,
        circle_tool(X, A),
        B,
        perp_bisector_tool(A, B)
    ]
