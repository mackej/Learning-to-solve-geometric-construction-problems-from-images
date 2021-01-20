from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction


def init(env):
    A,B,_ = env.add_free_segment(
        (227.5, 260.5), (312.0, 174.5))
    C = env.add_free(362.0, 277.5)

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection", "Parallel",
        "Compass"
    )
    env.goal_params(A,B,C)

def construct_goals(A,B,C):
    result = []
    v_base = B.a-A.a
    for v in v_base, -v_base:
        D = Point(C.a + v)
        result.append((D, segment_tool(C,D)))
    return result

def get_construction(env, obj):
    A, B, C = [obj[i] for i in env.goal_par_indices]
    goal = env.goals[env.goal_index]
    goal_p = goal[0]
    AB_half = Point((A.a + B.a )/2)
    c = compass_tool(A, B, C)
    p = parallel_tool(line_tool(A, B), C)
    X = goal_p
    return [
               construction.ConstructionProcess('Compass', [A, B, C]),
               construction.ConstructionProcess('Parallel', [AB_half, C]),
               construction.ConstructionProcess('Line', [X, C]),
               construction.ConstructionProcess('Line', [X, C]),
           ], [
               c,
               p,
               #X, line_tool(X, C)
           ]
