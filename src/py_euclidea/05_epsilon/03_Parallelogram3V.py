from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    A = env.add_free(299.5, 194.5)
    B = env.add_free(238.0, 285.0)
    C = env.add_free(383.0, 284.5)

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection", "Parallel"
    )
    env.goal_params(A, B, C)

def construct_goals(A, B, C):
    D, *segments = parallelogram(A, B, C)
    return segments

def get_construction(env, obj):
    A, B, C = [obj[i] for i in env.goal_par_indices]
    l = line_tool(A, B)
    l2 = line_tool(B, C)

    return [
               construction.ConstructionProcess('Line', [A, B]),
               construction.ConstructionProcess('Parallel', [Point((A.a + B.a)/2), C]),
               construction.ConstructionProcess('Line', [B, C]),
               construction.ConstructionProcess('Parallel', [Point((B.a + C.a) / 2), A]),
           ], [
               l,
               Point((A.a + B.a) / 2),
               parallel_tool(l, C),
               Point((B.a + C.a) / 2),
               l2,
               parallel_tool(l2, A)
           ]
