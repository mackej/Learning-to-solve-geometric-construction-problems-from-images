from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    (A,B,C,D),_ = env.add_free_trapezoid(
        (477.5, 331.5), (181.5, 331.5), (269.0, 213.0), (432.0, 213.0))

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection",
    )
    env.goal_params(A, B, C, D)

def construct_goals(A, B, C, D):
    X = (A.a+B.a)/2
    Y = (C.a+D.a)/2
    return Segment(X, Y)

def get_construction(env, obj):
    A,B,C,D = [obj[i] for i in env.goal_par_indices]
    X = Point((A.a + B.a) / 2)
    Y = Point((C.a + D.a) / 2)
    return [

        construction.ConstructionProcess('Perpendicular_Bisector', [A, B]),
        construction.ConstructionProcess('Perpendicular_Bisector', [C, D]),
        construction.ConstructionProcess('Line', [X, Y]),
        construction.ConstructionProcess('Line', [X, Y]),
        construction.ConstructionProcess('Line', [X, Y]),
    ], [
        perp_bisector_tool(A, B),
        perp_bisector_tool(C, D),
        X,
        Y,
        line_tool(X, Y)
    ]
