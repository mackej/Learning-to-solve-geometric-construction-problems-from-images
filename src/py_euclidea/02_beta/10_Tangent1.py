from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction
import numpy as np

def init(env):
    C = env.add_free(322.0, 253.5)
    X = env.add_free(420.5, 215.5)
    env.add_circle(C, X)

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "intersection", "Perpendicular",
    )
    env.goal_params(C, X)

def construct_goals(C, X):
    n = C.a - X.a
    return Line(n, np.dot(n, X.a))


def get_construction(env, obj):
    C = obj[0]
    X = obj[1]
    l = line_tool(C, X)
    return [
        construction.ConstructionProcess('Line', [C, X]),
        construction.ConstructionProcess('Perpendicular', [Point((C.a + X.a) / 2), X]),
    ], [
        l,
        perp_tool(l, X)
    ]
