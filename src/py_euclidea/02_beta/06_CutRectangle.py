from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction
import numpy as np


def init(env):
    (A,B,C,D), _ = env.add_free_rectangle(
        (433.0, 345.5), (229.0, 345.5), (229.0, 217.0))
    X = env.add_free(511.0, 113.0)

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "intersection",
    )
    env.goal_params(A, B, C, D, X)

def construct_goals(A, B, C, D, X):
    Y = Point((A.a+B.a+C.a+D.a)/4)
    return line_tool(X,Y)

def get_construction(env, obj):
    A = obj[0]
    B = obj[1]
    C = obj[4]
    D = obj[5]
    X = obj[10]
    Y = intersection_tool(line_tool(A,C),line_tool(B,D))
    return [
        construction.ConstructionProcess('Line', [A, C]),
        construction.ConstructionProcess('Line', [B, D]),
        construction.ConstructionProcess('Line', [X, Y]),
        construction.ConstructionProcess('Line', [X, Y]),
    ], [
        line_tool(A, C),
        line_tool(B, D),
        Y,
        line_tool(X,Y)
    ]
