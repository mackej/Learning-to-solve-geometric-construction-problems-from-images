from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    A,B,_ = env.add_free_segment(
        (169.0, 250.0), (270.5, 136.0))
    C = env.add_free(320.0, 239.5)

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection", "Parallel"
    )
    env.goal_params(A,B,C)

def construct_goals(A,B,C):
    A2 = Point(2*C.a-A.a)
    B2 = Point(2*C.a-B.a)
    return (
        A2, B2, segment_tool(A2, B2)
    )

def get_construction(env, obj):
    A, B, C = [obj[i] for i in env.goal_par_indices]
    l1 = line_tool(A, C)
    l2 = line_tool(B, C)
    c1 = circle_tool(C, B)
    B2 = Point(2*C.a-B.a)
    A2 = Point(2 * C.a - A.a)
    AB_half = Point((A.a + B.a)/2)
    p = parallel_tool(line_tool(A, B), B2)
    return [
               construction.ConstructionProcess('Line', [A, C]),
               construction.ConstructionProcess('Line', [B, C]),
               construction.ConstructionProcess('Circle', [C, B]),
               construction.ConstructionProcess('Parallel', [AB_half, B2]),
               construction.ConstructionProcess('Parallel', [AB_half, B2]),
               construction.ConstructionProcess('Point', [A2]),
           ], [
               l1, l2,
               c1,
               B2,
               p, A2
           ]
