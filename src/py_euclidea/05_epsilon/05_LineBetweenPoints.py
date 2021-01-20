from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    C = env.add_free(267.5, 174.5)
    X = env.add_free(426.0, 258.0)
    Y = env.add_free(327.5, 316.0)
    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection", "Parallel"
    )
    env.goal_params(C, X, Y)

def construct_goals(C, X, Y):
    D = Point(X.a+Y.a-C.a)
    return line_tool(C, D)

def get_construction(env, obj):
    A, B, C = [obj[i] for i in env.goal_par_indices]
    l = line_tool(B, C)
    p = perp_bisector_tool(B, C)
    P = intersection_tool(p, l)

    return [
               construction.ConstructionProcess('Line', [B, C]),
               construction.ConstructionProcess('Perpendicular_Bisector', [B, C]),
               construction.ConstructionProcess('Line', [P, A]),
               construction.ConstructionProcess('Line', [P, A]),
           ], [
               l,
               p,
               P,
               line_tool(P,A)
           ]
