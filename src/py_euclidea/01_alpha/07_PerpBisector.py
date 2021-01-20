from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction
import numpy as np

def init(env):
    A,B,_ = env.add_free_segment((247.4, 242.5), (405.5, 241))
    env.set_tools("move", "Point", "Line", "Circle", "intersection")
    env.goal_params(A, B)

def construct_goals(A, B):
    return (
        perp_bisector_tool(A, B),
    )

def get_construction(env, obj):
    A = obj[0]
    B = obj[1]
    C0, C1 = intersection_tool(
        circle_tool(A, B),
        circle_tool(B, A),
    )
    return [
        construction.ConstructionProcess('Circle', [A, B]),
        construction.ConstructionProcess('Circle', [B, A]),
        #construction.ConstructionProcess('Point', [C0]),
        #construction.ConstructionProcess('Point', [C1]),
        construction.ConstructionProcess('Line', [C1, C0]),
        construction.ConstructionProcess('Line', [C1, C0]),
        construction.ConstructionProcess('Line', [C1, C0]),
    ], [
        circle_tool(A, B),
        circle_tool(B, A),
        C0,
        C1,
        line_tool(C1, C0)
    ]
