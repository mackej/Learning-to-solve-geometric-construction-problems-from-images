from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction
import numpy as np

def init(env):
    (A,B,C,D),_ = env.add_free_square((218, 351), (419, 351.5))
    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "intersection",
    )
    env.goal_params(A, B, C, D)

def construct_goals(A, B, C, D):
    center = (A.a + C.a)/2
    radius = np.linalg.norm(B.a-A.a)/2
    return Circle(center, radius)

def get_construction(env, obj):
    A = obj[0]
    B = obj[1]
    C = obj[2]
    D = obj[3]
    center = Point((A.a + C.a) / 2)
    rad_point = Point((A.a + B.a) / 2)
    return [
        construction.ConstructionProcess('Line', [A, C]),
        construction.ConstructionProcess('Perpendicular_Bisector', [A, B]),
        #construction.ConstructionProcess('Point', [rad_point]),
        #construction.ConstructionProcess('Point', [center]),
        construction.ConstructionProcess('Circle', [center,rad_point]),
        construction.ConstructionProcess('Circle', [center, rad_point]),
        construction.ConstructionProcess('Circle', [center, rad_point]),
    ], [
        line_tool(A, C),
        perp_bisector_tool(A, B),
        center,
        rad_point,
        circle_tool(center, rad_point)
    ]
