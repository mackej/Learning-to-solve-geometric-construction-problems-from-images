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
    D = Point(C.a + Y.a - X.a)
    return line_tool(C, D)

def get_construction(env, obj):
    A, B, C = [obj[i] for i in env.goal_par_indices]
    l = line_tool(B, C)

    return [
               construction.ConstructionProcess('Line', [B, C]),
               construction.ConstructionProcess('Parallel', [Point((B.a + C.a)/2), A]),
           ], [
               l,
               parallel_tool(l, A),
           ]

