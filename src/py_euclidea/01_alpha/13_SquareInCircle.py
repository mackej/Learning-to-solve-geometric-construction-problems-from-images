from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction
import numpy as np

def init(env):
    O = env.add_free(319.5, 245)
    P = env.add_free(312.5, 125.5)
    env.add_circle(O, P)
    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "intersection",
    )
    env.goal_params(O, P)

def construct_goals(O, P):
    vertices = []
    v = P.a - O.a
    for _ in range(3):
        v = vector_perp_rot(v)
        vertices.append(Point(O.a + v))
    result = [
        segment_tool(X,Y)
        for X,Y in zip([P]+vertices, vertices+[P])
    ]
    return result

def get_construction(env, obj):
    O = obj[0]
    P = obj[1]
    circle = obj[2]
    P1,P2 = intersection_tool(line_tool(O, P), circle)
    if same_point(P1.a, P.a):
        C = P2
    else:
        C = P1
    B, D = intersection_tool(perp_bisector_tool(P,C), circle)
    return [
        construction.ConstructionProcess('Line', [O, P]),
        # construction.ConstructionProcess('Point', [C]),
        construction.ConstructionProcess('Perpendicular_Bisector', [P, C]),
        construction.ConstructionProcess('Perpendicular_Bisector', [P, C]),

        # construction.ConstructionProcess('Point', [B]),
        # construction.ConstructionProcess('Point', [D]),
        construction.ConstructionProcess('Line', [P, B]),
        construction.ConstructionProcess('Line', [P, B]),
        construction.ConstructionProcess('Line', [B, C]),
        construction.ConstructionProcess('Line', [C, D]),
        construction.ConstructionProcess('Line', [C, D]),
        construction.ConstructionProcess('Line', [P, D]),
    ], [
        line_tool(O, P),
        C,
        perp_bisector_tool(P, C),
        B,
        D,
        line_tool(P, B),
        line_tool(B, C),
        line_tool(C, D),
        line_tool(P, D)
    ]
