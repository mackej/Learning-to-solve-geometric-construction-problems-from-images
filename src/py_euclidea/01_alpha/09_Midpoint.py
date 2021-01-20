from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction
import numpy as np

def init(env):
    A = env.add_free(263, 285)
    B = env.add_free(398.5, 220.5)
    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "intersection",
    )
    env.goal_params(A, B)

def construct_goals(A, B):
    return (
        Point((A.a + B.a)/2)
    )

def get_construction(env, obj):
    A = obj[0]
    B = obj[1]
    return [
        construction.ConstructionProcess('Perpendicular_Bisector', [A, B]),
        construction.ConstructionProcess('Line', [A, B]),
        construction.ConstructionProcess('Point', [Point((A.a + B.a) / 2)]),
    ], [
        perp_bisector_tool(A, B),
        line_tool(A, B),
        Point((A.a + B.a) / 2)
    ]
