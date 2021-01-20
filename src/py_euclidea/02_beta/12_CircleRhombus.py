from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction
import numpy as np

def init(env):
    A = env.add_free(170.5, 249.5)
    C = env.add_free(472.5, 247.0)
    axis = env.add_constr(
        perp_bisector_tool, (A, C), Line,
        hidden = True,
    )
    B = env.add_dep((321.0, 171.0), axis)
    D, *segments = env.add_constr(
        parallelogram, (A,B,C),
        (Point, Line, Line, Line, Line),
    )

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "intersection", "Perpendicular",
    )
    env.goal_params(A, C, segments[0])

def construct_goals(A, C, seg0):
    center = (A.a+C.a)/2
    return Circle(center, seg0.dist_from(center))

def get_construction(env, obj):
    A = obj[0]
    C = obj[1]
    B = obj[3]
    D = obj[4]
    X = intersection_tool( line_tool(A,C), line_tool(B,D))
    AB = Point((A.a + B.a)/2)
    V = intersection_tool(perp_tool(line_tool(A,B),X),line_tool(A,B))
    return [
        construction.ConstructionProcess('Line', [A, C]),
        construction.ConstructionProcess('Line', [B, D]),
        construction.ConstructionProcess('Perpendicular', [AB, X]),
        construction.ConstructionProcess('Perpendicular', [AB, X]),
        construction.ConstructionProcess('Circle', [X, V]),
        construction.ConstructionProcess('Circle', [X, V]),
    ], [
        line_tool(A, C),
        line_tool(B, D),
        X,
        perp_tool(line_tool(A,B), X),
        V,
        circle_tool(X, V)
    ]
