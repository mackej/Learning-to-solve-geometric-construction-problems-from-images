from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction
import numpy as np

def init(env):
    X_x = 50.0
    X_y = 50.0
    l_p1 = (100.0, 0.0)
    l_p2 = (0.0, 0.0)
    X = env.add_free(X_x, X_y)
    l = env.add_free_line(l_p1, l_p2)
    X_Point = Point([X_x, X_y])
    l_line = line_tool(Point(l_p1), Point(l_p2))
    p = perp_tool(l_line, X_Point)
    Cross = intersection_tool(p, l_line)
    #env.add_free(Cross.a, hidden=True)
    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "intersection",
    )
    env.goal_params(X, l)

def construct_goals(X, l):
    return perp_tool(l, X)


def additional_bb(X, l, goal):
    p  = intersection_tool(goal,l)
    return p

def get_construction(env, obj):
    g = env.cur_goal()[0]
    Y = obj[0]
    l = obj[3]
    X = intersection_tool(g, l)
    rnd_pt = Point(X.a + (X.a-Y.a)/2)
    A, B = intersection_tool(circle_tool(Y, rnd_pt), l)
    return [
        construction.ConstructionProcess('Circle', [Y, rnd_pt]),
        construction.ConstructionProcess('Circle', [Y, rnd_pt]),
        construction.ConstructionProcess('Perpendicular_Bisector', [A, B]),
        construction.ConstructionProcess('Perpendicular_Bisector', [A, B]),
        construction.ConstructionProcess('Perpendicular_Bisector', [A, B]),
    ], [
        rnd_pt,
        circle_tool(Y, rnd_pt),
        A,
        B,
        perp_bisector_tool(A, B)
    ]
