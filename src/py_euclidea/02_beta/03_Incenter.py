from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction
import numpy as np

def init(env):
    A = env.add_free(178.0, 357.5)
    B = env.add_free(466.0, 357.0)
    C = env.add_free(274.5, 136.5)
    env.add_segment(A, B)
    env.add_segment(B, C)
    env.add_segment(C, A)
    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "intersection",
    )
    env.goal_params(A, B, C)

def construct_goals(A, B, C):
    a,b,c = (
        np.linalg.norm(Y.a-X.a)
        for (X,Y) in ((B,C), (C,A), (A,B))
    )
    return Point((a*A.a + b*B.a + c*C.a)/(a+b+c))

def get_construction(env, obj):
    A = obj[0]
    B = obj[1]
    C= obj[2]
    g = env.cur_goal()

    return [
        construction.ConstructionProcess('Angle_Bisector', [A, B, C]),
        construction.ConstructionProcess('Angle_Bisector', [A, C, B]),
        construction.ConstructionProcess('Point', [Point(g[0].a)]),
    ], [
        angle_bisector_tool(A, B, C),
        angle_bisector_tool(A, C, B),
        Point(g[0].a),
    ]
