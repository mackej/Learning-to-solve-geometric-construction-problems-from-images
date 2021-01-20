from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction


def init(env):
    A,B,_ = env.add_free_segment(
        (173.5, 306.0), (386.0, 305.5))
    M = env.add_free(367.0, 174.0)

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection", "Parallel",
        "Compass"
    )
    env.goal_params(A,B,M)

def construct_goals(A,B,M):
    v = (B.a-A.a)/2
    C = Point(M.a+v)
    D = Point(M.a-v)
    return (
        segment_tool(B,C),
        segment_tool(C,D),
        segment_tool(D,A),
    )
def get_construction(env, obj):
    A, B, M = [obj[i] for i in env.goal_par_indices]
    ab = line_tool(A, B)
    AB_half = Point((A.a + B.a) / 2)
    goal = env.goals[env.goal_index]
    v = perp_bisector_tool(A, B)
    M2 = intersection_tool(v, ab)
    center = Point((M.a+M2.a)/2)
    p1 = parallel_tool(v, A)
    p2 = parallel_tool(v, B)

    return [
               construction.ConstructionProcess('Parallel', [AB_half, M]),
               construction.ConstructionProcess('Perpendicular_Bisector', [A, B]),
               construction.ConstructionProcess('Line', [M2, M]),
               construction.ConstructionProcess('Line', [M2, M]),
               construction.ConstructionProcess('Parallel', [center, A]),
               construction.ConstructionProcess('Parallel', [center, B]),
           ], [
               v,
               Point((B.a - A.a) / 4),
               M2,
               p1,
               p2,
           ]
'''
def get_construction(env, obj):
    A, B, M = [obj[i] for i in env.goal_par_indices]
    ab = line_tool(A, B)
    goal = env.goals[env.goal_index]
    v = perp_bisector_tool(A, B)
    M2 = intersection_tool(v, ab)
    c = compass_tool(M2, A, M)
    p = parallel_tool(ab, M)
    v_x = (B.a - A.a) / 2
    C = Point(M.a + v_x)
    D = Point(M.a - v_x)
    AB_half = Point((A.a + B.a)/2)

    return [
               construction.ConstructionProcess('Parallel', [AB_half, M]),
               construction.ConstructionProcess('Perpendicular_Bisector', [A, B]),
               construction.ConstructionProcess('Compass', [M2, A, M]),
               construction.ConstructionProcess('Compass', [M2, A, M]),
               construction.ConstructionProcess('Line', [A, D]),
               construction.ConstructionProcess('Line', [A, D]),
               construction.ConstructionProcess('Line', [B, C]),
               construction.ConstructionProcess('Line', [B, C]),
           ], [
               v,
               Point((B.a - A.a) / 4),
               M2,
               c,
               p,
               C, D,
               line_tool(A, D), line_tool(B, C)
           ]
'''
